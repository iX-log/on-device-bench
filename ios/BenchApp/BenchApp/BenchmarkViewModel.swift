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

@MainActor
@Observable
final class BenchmarkViewModel {
	private(set) var log = "ready"
	private(set) var progress = ""
	private(set) var isRunning = false

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

			let url = try RunWriter.write(samples: samples, duration: seconds)
			self.log = session.header
				+ SustainedSummary.text(samples: samples, file: url.lastPathComponent)
			self.progress = ""
		}
	}

	// MARK: - Internals

	private struct Session {
		let model: whisper_base_encoder_fp16
		let input: MLMultiArray
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

		let t0 = CACurrentMediaTime()
		let model = try whisper_base_encoder_fp16(configuration: config)
		let loadMs = (CACurrentMediaTime() - t0) * 1000

		let after = Memory.physFootprint()
		out += "after load:    \(Memory.mb(after))\n"
		if let a = before, let b = after { out += "model cost:    \(Memory.mb(b - a))\n" }
		out += String(format: "load:          %.1f ms\n\n", loadMs)

		return Session(model: model, input: try Self.makeInput(), header: out)
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
		_ = try s.model.prediction(mel: s.input)
		return (CACurrentMediaTime() - t) * 1000
	}
}
