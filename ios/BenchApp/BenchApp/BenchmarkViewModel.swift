//
//  BenchmarkViewModel.swift
//  BenchApp
//
//  Created by Logan on 07.09.26.
//

import Foundation
import Observation
import CoreML
import UIKit
import QuartzCore

enum Precision: String, CaseIterable, Identifiable {
	case fp16, int8, int4

	var id: String { rawValue }
	var label: String { rawValue }
}

@MainActor
@Observable
final class BenchmarkViewModel {
	private(set) var log = "ready"
	private(set) var progress = ""
	private(set) var isRunning = false
	var selected: Precision = .fp16

	var selectedModelInfo: String {
		let url = Self.modelURL(for: selected)
		return "\(url.lastPathComponent) — \(Memory.mb(Self.directorySize(at: url)))"
	}

	// MARK: - Public API

	func runQuick(iterations: Int = 100) async {
		await guarded {
			let session = try self.loadModel()
			self.log = session.header

			var times: [Double] = []
			for _ in 0..<iterations {
				times.append(try self.timeOneInference(session))
			}

			// Measured warm-up is exactly one run — see RESULTS.md.
			let warm = Array(times.dropFirst()).sorted()

			var out = session.header
			out += String(format: "run 1 (cold):  %.1f ms\n", times[0])
			out += String(format: "min:           %.1f ms\n", warm.first ?? 0)
			out += String(format: "median:        %.1f ms\n", Stats.percentile(warm, 0.50))
			out += String(format: "p95:           %.1f ms\n", Stats.percentile(warm, 0.95))
			out += String(format: "max:           %.1f ms\n", warm.last ?? 0)
			out += "\npeak footprint: \(Memory.mb(Memory.physFootprint()))\n"
			out += "available now:  \(Memory.mb(Memory.available()))\n"
			self.log = out
		}
	}

	func runSustained(seconds: Double = 600) async {
		// Screen must stay on or iOS suspends us. Adds heat — note as a limitation.
		UIApplication.shared.isIdleTimerDisabled = true
		defer { UIApplication.shared.isIdleTimerDisabled = false }

		await guarded {
			let session = try self.loadModel()
			self.log = session.header + "sustained run started…\n"

			var samples: [Sample] = []
			let start = CACurrentMediaTime()
			var i = 0

			while CACurrentMediaTime() - start < seconds {
				let latency = try self.timeOneInference(session)
				let elapsed = CACurrentMediaTime() - start

				samples.append(Sample(
					index: i,
					elapsed_s: elapsed,
					latency_ms: latency,
					thermal: ProcessInfo.processInfo.thermalState.name,
					footprint_bytes: Memory.physFootprint() ?? 0
				))
				i += 1

				if i % 200 == 0 {
					self.progress = String(
						format: "%.0fs / %.0fs   n=%d   last %.1f ms   %@",
						elapsed, seconds, i, latency, samples.last!.thermal
					)
					await Task.yield()
				}
			}

			let url = try RunWriter.write(samples: samples, duration: seconds, precision: self.selected)
			self.log = session.header
				+ SustainedSummary.text(samples: samples, file: url.lastPathComponent)
			self.progress = ""
		}
	}

	// MARK: - Internals

	private struct Session {
		/// Abstracts over the three generated model classes, which share no
		/// base class or protocol — a closure sidesteps that instead of one.
		let predict: () throws -> Void
		let header: String
	}

	private func guarded(_ body: @escaping () async throws -> Void) async {
		isRunning = true
		defer { isRunning = false }
		do { try await body() }
		catch { log = "error: \(error)" }
	}

	private func loadModel() throws -> Session {
		var out = ""
		let before = Memory.physFootprint()
		out += "before load:   \(Memory.mb(before))\n"
		out += "available:     \(Memory.mb(Memory.available()))\n"

		let config = MLModelConfiguration()
		config.computeUnits = .all
		let input = try Self.makeInput()

		let t0 = CACurrentMediaTime()
		let predict: () throws -> Void
		switch selected {
		case .fp16:
			let model = try whisper_base_encoder_fp16(configuration: config)
			predict = { _ = try model.prediction(mel: input) }
		case .int8:
			let model = try whisper_base_encoder_int8(configuration: config)
			predict = { _ = try model.prediction(mel: input) }
		case .int4:
			let model = try whisper_base_encoder_int4(configuration: config)
			predict = { _ = try model.prediction(mel: input) }
		}
		let loadMs = (CACurrentMediaTime() - t0) * 1000
		let modelURL = Self.modelURL(for: selected)

		let after = Memory.physFootprint()
		out += "after load:    \(Memory.mb(after))\n"
		if let a = before, let b = after { out += "model cost:    \(Memory.mb(b - a))\n" }
		out += String(format: "load:          %.1f ms\n", loadMs)
		out += "model file:    \(modelURL.lastPathComponent)\n"
		out += "on disk:       \(Memory.mb(Self.directorySize(at: modelURL)))\n\n"

		return Session(predict: predict, header: out)
	}

	private static func modelURL(for precision: Precision) -> URL {
		switch precision {
		case .fp16: whisper_base_encoder_fp16.urlOfModelInThisBundle
		case .int8: whisper_base_encoder_int8.urlOfModelInThisBundle
		case .int4: whisper_base_encoder_int4.urlOfModelInThisBundle
		}
	}

	private static func directorySize(at url: URL) -> UInt64 {
		guard let enumerator = FileManager.default.enumerator(
			at: url, includingPropertiesForKeys: [.fileSizeKey]
		) else { return 0 }

		var total: UInt64 = 0
		for case let fileURL as URL in enumerator {
			let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
			total += UInt64(size)
		}
		return total
	}

	/// Raw pointer fill — NSNumber subscripting 240k times would cost
	/// more than the inference we're trying to measure.
	private static func makeInput() throws -> MLMultiArray {
		let mel = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
		let ptr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
		for i in 0..<mel.count { ptr[i] = Float.random(in: -1...1) }
		return mel
	}

	private func timeOneInference(_ s: Session) throws -> Double {
		let t = CACurrentMediaTime()
		try s.predict()
		return (CACurrentMediaTime() - t) * 1000
	}
}
