//
//  Support.swift
//  BenchApp
//
//  Created by Logan on 07.09.26.
//

import Foundation
import UIKit

// MARK: - Memory

enum Memory {
	/// Bytes the kernel charges to this process — what jetsam uses.
	/// Not `resident_size`, which misses IOKit (ANE) allocations.
	static func physFootprint() -> UInt64? {
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

	static func available() -> UInt64 { UInt64(os_proc_available_memory()) }

	static func mb(_ bytes: UInt64?) -> String {
		guard let b = bytes else { return "n/a" }
		return String(format: "%.1f MB", Double(b) / 1_048_576)
	}
}

extension ProcessInfo.ThermalState {
	var name: String {
		switch self {
		case .nominal: "nominal"
		case .fair: "fair"
		case .serious: "serious"
		case .critical: "critical"
		@unknown default: "unknown"
		}
	}
}

// MARK: - Stats

enum Stats {
	/// Expects a sorted array.
	static func percentile(_ sorted: [Double], _ p: Double) -> Double {
		guard !sorted.isEmpty else { return 0 }
		return sorted[Int((Double(sorted.count - 1) * p).rounded())]
	}

	static func median(_ values: [Double]) -> Double {
		percentile(values.sorted(), 0.50)
	}
}

// MARK: - Records

struct Sample: Codable {
	let index: Int
	let elapsed_s: Double
	let latency_ms: Double
	let thermal: String
	let footprint_bytes: UInt64
}

struct RunFile: Codable {
	let started_at: String
	let device: String
	let os_version: String
	let model: String
	let precision: String
	let duration_target_s: Double
	let notes: String
	let samples: [Sample]
}

enum RunWriter {
	static func write(samples: [Sample], duration: Double, precision: Precision) throws -> URL {
		let file = RunFile(
			started_at: ISO8601DateFormatter().string(from: Date()),
			device: "iPhone 14 Pro Max (A16, 6GB)",
			os_version: UIDevice.current.systemVersion,
			model: "whisper-base encoder",
			precision: precision.rawValue,
			duration_target_s: duration,
			notes: "airplane mode, off power, screen on min brightness, idle timer disabled",
			samples: samples
		)
		let enc = JSONEncoder()
		enc.outputFormatting = .prettyPrinted
		let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
		let url = dir.appendingPathComponent("sustained-\(precision.rawValue)-\(Int(Date().timeIntervalSince1970)).json")
		try enc.encode(file).write(to: url)
		return url
	}
}

// MARK: - Summary

enum SustainedSummary {
	static func text(samples: [Sample], file: String) -> String {
		guard let last = samples.last else { return "no samples\n" }
		let sorted = samples.map(\.latency_ms).sorted()
		let firstMin = samples.filter { $0.elapsed_s < 60 }.map(\.latency_ms)
		let lastMin = samples.filter { $0.elapsed_s > last.elapsed_s - 60 }.map(\.latency_ms)

		var out = ""
		out += "inferences:    \(samples.count)\n"
		out += String(format: "elapsed:       %.1f s\n\n", last.elapsed_s)
		out += String(format: "median all:    %.1f ms\n", Stats.percentile(sorted, 0.50))
		out += String(format: "p95 all:       %.1f ms\n", Stats.percentile(sorted, 0.95))
		out += String(format: "first minute:  %.1f ms\n", Stats.median(firstMin))
		out += String(format: "last minute:   %.1f ms\n", Stats.median(lastMin))

		let a = Stats.median(firstMin), b = Stats.median(lastMin)
		out += String(format: "drift:         %+.1f ms (%+.0f%%)\n\n", b - a, (b - a) / a * 100)

		var seen: String?
		for s in samples where s.thermal != seen {
			out += String(format: "  %6.1fs → %@\n", s.elapsed_s, s.thermal)
			seen = s.thermal
		}
		return out + "\nwrote \(file)\n"
	}
}
