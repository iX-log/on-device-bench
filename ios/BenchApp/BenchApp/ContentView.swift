import SwiftUI
import CoreML

// MARK: - Memory

/// Bytes the kernel charges to this process. This is the number jetsam
/// uses when deciding what to kill, and what Xcode's memory gauge shows.
/// Not `resident_size` — that misses IOKit allocations (where ANE buffers
/// live) and includes shared framework pages that aren't really ours.
func physFootprint() -> UInt64? {
	var info = task_vm_info_data_t()
	var count = mach_msg_type_number_t(
		MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
	)
	let kr = withUnsafeMutablePointer(to: &info) {
		$0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
			task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
		}
	}
	return kr == KERN_SUCCESS ? info.phys_footprint : nil
}

/// Bytes remaining before jetsam. Live estimate, moves with system pressure.
func availableMemory() -> UInt64 {
	UInt64(os_proc_available_memory())
}

func mb(_ bytes: UInt64?) -> String {
	guard let b = bytes else { return "n/a" }
	return String(format: "%.1f MB", Double(b) / 1_048_576)
}

// MARK: - Stats

func percentile(_ sorted: [Double], _ p: Double) -> Double {
	guard !sorted.isEmpty else { return 0 }
	let idx = Int((Double(sorted.count - 1) * p).rounded())
	return sorted[idx]
}

// MARK: - View

struct ContentView: View {
	@State private var log = "ready"
	@State private var running = false

	var body: some View {
		ScrollView {
			Text(log).monospaced().font(.caption).padding()
		}
		.safeAreaInset(edge: .bottom) {
			Button(running ? "running…" : "Run") {
				Task { await run() }
			}
			.disabled(running)
			.padding()
		}
	}

	func run() async {
		running = true
		defer { running = false }

		var out = ""
		do {
			let memBefore = physFootprint()
			out += "before load:   \(mb(memBefore))\n"
			out += "available:     \(mb(availableMemory()))\n"

			let config = MLModelConfiguration()
			config.computeUnits = .all

			let t0 = CACurrentMediaTime()
			let model = try whisper_base_encoder_fp16(configuration: config)
			let loadMs = (CACurrentMediaTime() - t0) * 1000

			let memAfterLoad = physFootprint()
			out += "after load:    \(mb(memAfterLoad))\n"
			if let a = memBefore, let b = memAfterLoad {
				out += "model cost:    \(mb(b - a))\n"
			}
			out += String(format: "load:          %.1f ms\n\n", loadMs)

			// Input buffer, filled via raw pointer — NSNumber subscripting
			// 240k times would cost more than the inference itself.
			let mel = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
			let ptr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
			for i in 0..<mel.count { ptr[i] = Float.random(in: -1...1) }

			// Poll memory during inference: phys_footprint is point-in-time,
			// so a transient spike is invisible to before/after reads.
			let peak = MemoryPeak()
			peak.start()

			var times: [Double] = []
			for _ in 1...100 {
				let t = CACurrentMediaTime()
				_ = try model.prediction(mel: mel)
				times.append((CACurrentMediaTime() - t) * 1000)
			}

			let peakBytes = peak.stop()

			// Discard run 1 — measured warm-up is exactly one run.
			let warm = Array(times.dropFirst()).sorted()

			out += String(format: "run 1 (cold):  %.1f ms\n", times[0])
			out += String(format: "min:           %.1f ms\n", warm.first ?? 0)
			out += String(format: "median:        %.1f ms\n", percentile(warm, 0.50))
			out += String(format: "p95:           %.1f ms\n", percentile(warm, 0.95))
			out += String(format: "max:           %.1f ms\n", warm.last ?? 0)
			out += "\npeak footprint: \(mb(peakBytes))\n"
			out += "available now:  \(mb(availableMemory()))\n"

			log = out
		} catch {
			log = "error: \(error)"
		}
	}
}

// MARK: - Peak sampler

/// Samples phys_footprint on a background thread and keeps the maximum.
/// Sampling has its own cost — note it in limitations.
final class MemoryPeak {
	private var timer: DispatchSourceTimer?
	private var peak: UInt64 = 0
	private let queue = DispatchQueue(label: "mem.peak")

	func start() {
		peak = physFootprint() ?? 0
		let t = DispatchSource.makeTimerSource(queue: queue)
		t.schedule(deadline: .now(), repeating: .milliseconds(10))
		t.setEventHandler { [weak self] in
			guard let self, let now = physFootprint() else { return }
			if now > self.peak { self.peak = now }
		}
		t.resume()
		timer = t
	}

	func stop() -> UInt64 {
		timer?.cancel()
		timer = nil
		return queue.sync { peak }
	}
}
