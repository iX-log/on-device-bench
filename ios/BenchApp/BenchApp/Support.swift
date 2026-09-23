//
//  Support.swift
//  BenchApp
//
//  Created by Logan on 07.09.26.
//

import Foundation
import UIKit
import Network

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

extension UIDevice.BatteryState {
	var name: String {
		switch self {
		case .unknown: "unknown"
		case .unplugged: "unplugged"
		case .charging: "charging"
		case .full: "full"
		@unknown default: "unknown"
		}
	}
}

// MARK: - Device conditions

/// iOS never exposes airplane-mode state to apps. `NWPathMonitor` reporting
/// no available interface is a proxy for it, not a direct read -- airplane
/// mode with Wi-Fi switched back on, for instance, would show as available
/// here.
enum NetworkCheck {
	static func anyInterfaceAvailable() async -> Bool {
		await withCheckedContinuation { continuation in
			let monitor = NWPathMonitor()
			let queue = DispatchQueue(label: "on-device-bench.network-check")
			var resumed = false
			monitor.pathUpdateHandler = { path in
				guard !resumed else { return }
				resumed = true
				continuation.resume(returning: path.status == .satisfied)
				monitor.cancel()
			}
			monitor.start(queue: queue)
		}
	}
}

/// Everything about the device and its state captured once, at the start of
/// a run, so RunFile records what was actually true instead of a hardcoded
/// guess.
struct DeviceSnapshot: Codable {
	let device: String
	let os_version: String
	let physical_memory_bytes: UInt64
	let low_power_mode_enabled: Bool
	let battery_state: String
	let battery_level: Float
	let thermal_state_at_start: String
	let network_available_at_start: Bool

	var isThermalNominal: Bool { thermal_state_at_start == ProcessInfo.ThermalState.nominal.name }

	@MainActor
	static func capture() async -> DeviceSnapshot {
		UIDevice.current.isBatteryMonitoringEnabled = true
		return DeviceSnapshot(
			device: hardwareIdentifier(),
			os_version: UIDevice.current.systemVersion,
			physical_memory_bytes: ProcessInfo.processInfo.physicalMemory,
			low_power_mode_enabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
			battery_state: UIDevice.current.batteryState.name,
			battery_level: UIDevice.current.batteryLevel,
			thermal_state_at_start: ProcessInfo.processInfo.thermalState.name,
			network_available_at_start: await NetworkCheck.anyInterfaceAvailable()
		)
	}

	/// Raw `utsname().machine` value, e.g. "iPhone17,1" -- deliberately not
	/// mapped to a marketing name, since that mapping table goes stale with
	/// every new device Apple ships.
	private static func hardwareIdentifier() -> String {
		var info = utsname()
		uname(&info)
		return withUnsafePointer(to: &info.machine) {
			$0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
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

/// schema_version 2: `device`, `notes`, and the standing conditions used to
/// be string literals asserting what was true rather than reading it. See
/// `DeviceSnapshot`. Files written before this change have no
/// `schema_version` key at all -- treat its absence as version 1.
struct RunFile: Codable {
	let schema_version: Int
	let started_at: String
	let device: String
	let os_version: String
	let model: String
	let precision: String
	let duration_target_s: Double
	let physical_memory_bytes: UInt64
	let low_power_mode_enabled: Bool
	let battery_state: String
	let battery_level: Float
	let thermal_state_at_start: String
	let network_available_at_start: Bool
	let samples: [Sample]
}

enum RunWriter {
	static func write(samples: [Sample], duration: Double, precision: Precision, conditions: DeviceSnapshot) throws -> URL {
		let file = RunFile(
			schema_version: 2,
			started_at: ISO8601DateFormatter().string(from: Date()),
			device: conditions.device,
			os_version: conditions.os_version,
			model: "whisper-base encoder",
			precision: precision.rawValue,
			duration_target_s: duration,
			physical_memory_bytes: conditions.physical_memory_bytes,
			low_power_mode_enabled: conditions.low_power_mode_enabled,
			battery_state: conditions.battery_state,
			battery_level: conditions.battery_level,
			thermal_state_at_start: conditions.thermal_state_at_start,
			network_available_at_start: conditions.network_available_at_start,
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

// MARK: - Memory Ceiling

struct CeilingSample: Codable {
	let block_index: Int
	let allocated_bytes: UInt64
	let phys_footprint_bytes: UInt64
	let available_bytes: UInt64
	let elapsed_s: Double
}

struct CeilingProgress: Codable {
	let started_at: String
	let updated_at: String
	let chunk_bytes: UInt64
	let samples: [CeilingSample]
}

enum CeilingWriter {
	static let chunkBytes = 32 * 1024 * 1024

	static func fileURL() -> URL {
		FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
			.appendingPathComponent("ceiling-progress.json")
	}

	/// Overwrites the progress file and fsyncs before returning. The process
	/// gets jetsam-killed with no warning as it approaches the ceiling, so
	/// anything not physically on disk by the time this call returns is lost
	/// — `Data.write(atomically:)` alone doesn't guarantee that.
	static func flush(_ progress: CeilingProgress) {
		guard let data = try? JSONEncoder().encode(progress) else { return }
		let fd = open(fileURL().path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
		guard fd >= 0 else { return }
		defer { close(fd) }
		data.withUnsafeBytes { raw in
			_ = raw.baseAddress.map { write(fd, $0, raw.count) }
		}
		fsync(fd)
	}

	static func readLast() -> CeilingProgress? {
		guard let data = try? Data(contentsOf: fileURL()) else { return nil }
		return try? JSONDecoder().decode(CeilingProgress.self, from: data)
	}

	static func clear() {
		try? FileManager.default.removeItem(at: fileURL())
	}

	static func summaryText(_ progress: CeilingProgress) -> String {
		guard let last = progress.samples.last else { return "" }
		var out = "last memory ceiling probe — \(progress.updated_at)\n"
		out += "  blocks:      \(last.block_index + 1)  (\(Memory.mb(last.allocated_bytes)) allocated)\n"
		out += "  footprint:   \(Memory.mb(last.phys_footprint_bytes))\n"
		out += "  available:   \(Memory.mb(last.available_bytes))\n"
		return out
	}
}

// MARK: - Summary

enum SustainedSummary {
	static func text(samples: [Sample], file: String, conditions: DeviceSnapshot) -> String {
		guard let last = samples.last else { return "no samples\n" }
		let sorted = samples.map(\.latency_ms).sorted()
		let firstMin = samples.filter { $0.elapsed_s < 60 }.map(\.latency_ms)
		let lastMin = samples.filter { $0.elapsed_s > last.elapsed_s - 60 }.map(\.latency_ms)

		var out = ""
		if !conditions.isThermalNominal {
			out += "⚠️  started at thermalState \"\(conditions.thermal_state_at_start)\" — device wasn't cooled, this run isn't comparable to one that started nominal\n\n"
		}
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
