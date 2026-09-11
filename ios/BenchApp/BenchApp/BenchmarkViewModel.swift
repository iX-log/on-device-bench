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

enum BenchInputError: Error {
	case missingResource(String)
	case sizeMismatch(String)
}

@MainActor
@Observable
final class BenchmarkViewModel {
	private(set) var log = "ready"
	private(set) var progress = ""
	private(set) var isRunning = false
	var selected: Precision = .fp16
	var useRealInput = false

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

	/// Runs one inference per packed mel window in the bundle and writes each
	/// encoder output to Documents as `window-N-{precision}.bin` — flat
	/// little-endian float32, same convention as the mel .bin inputs.
	func dumpFeatures() async {
		await guarded {
			let filenames = Self.bundledPackedWindowFilenames()
			guard !filenames.isEmpty else {
				throw BenchInputError.missingResource("no window*.bin files found in bundle")
			}

			let config = MLModelConfiguration()
			config.computeUnits = .all
			let encode = try Self.loadEncoder(precision: self.selected, config: config)

			let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
			var out = "dumping features (\(self.selected.label)) ...\n"

			for filename in filenames {
				let input = try Self.makeInput(filename: filename)
				let hidden = try encode(input) // Float16 storage regardless of precision — upcast below.
				let shaped = MLShapedArray<Float>(converting: hidden)
				let data = shaped.withUnsafeShapedBufferPointer { ptr, _, _ in Data(buffer: ptr) }

				let index = Int(filename.dropFirst("window".count)) ?? 0
				let outName = "window-\(index)-\(self.selected.rawValue).bin"
				try data.write(to: docsDir.appendingPathComponent(outName))

				out += "  \(filename) -> \(outName)  (\(data.count) bytes)\n"
			}

			out += "\nwrote \(filenames.count) feature file(s) -> Documents/\n"
			self.log = out
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

		let inputName: String?
		if useRealInput {
			guard let name = Self.firstBundledMelFilename() else {
				throw BenchInputError.missingResource("no .bin files found in bundle mel/")
			}
			inputName = name
		} else {
			inputName = nil
		}
		let input = try Self.makeInput(filename: inputName)
		out += "input:         \(inputName.map { "real (\($0))" } ?? "synthetic")\n"

		let t0 = CACurrentMediaTime()
		let encode = try Self.loadEncoder(precision: selected, config: config)
		let loadMs = (CACurrentMediaTime() - t0) * 1000
		let predict: () throws -> Void = { _ = try encode(input) }
		let modelURL = Self.modelURL(for: selected)

		let after = Memory.physFootprint()
		out += "after load:    \(Memory.mb(after))\n"
		if let a = before, let b = after { out += "model cost:    \(Memory.mb(b - a))\n" }
		out += String(format: "load:          %.1f ms\n", loadMs)
		out += "model file:    \(modelURL.lastPathComponent)\n"
		out += "on disk:       \(Memory.mb(Self.directorySize(at: modelURL)))\n\n"

		return Session(predict: predict, header: out)
	}

	/// Instantiates the model for `precision` and returns a closure running one
	/// inference. Abstracts over the three generated model classes, which
	/// share no base class or protocol — see `Session`.
	private static func loadEncoder(precision: Precision, config: MLModelConfiguration) throws -> (MLMultiArray) throws -> MLMultiArray {
		switch precision {
		case .fp16:
			let model = try whisper_base_encoder_fp16(configuration: config)
			return { try model.prediction(mel: $0).hidden_states }
		case .int8:
			let model = try whisper_base_encoder_int8(configuration: config)
			return { try model.prediction(mel: $0).hidden_states }
		case .int4:
			let model = try whisper_base_encoder_int4(configuration: config)
			return { try model.prediction(mel: $0).hidden_states }
		}
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

	/// Raw pointer fill/copy — NSNumber subscripting 240k times would cost
	/// more than the inference we're trying to measure.
	///
	/// With `filename` nil, fills with random noise (synthetic). Otherwise
	/// loads `<filename>.bin` — a flat little-endian float32 dump from
	/// convert/export_mel_bin.py — via memcpy. The `mel/` source folder is a
	/// synchronized group, so Xcode flattens it into the bundle root rather
	/// than preserving it as a subdirectory — look up resources with no
	/// `subdirectory:` accordingly.
	private static func makeInput(filename: String? = nil) throws -> MLMultiArray {
		let mel = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
		let ptr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)

		guard let filename else {
			for i in 0..<mel.count { ptr[i] = Float.random(in: -1...1) }
			return mel
		}

		guard let url = Bundle.main.url(forResource: filename, withExtension: "bin") else {
			throw BenchInputError.missingResource("\(filename).bin")
		}
		let data = try Data(contentsOf: url)
		let expectedBytes = mel.count * MemoryLayout<Float>.size
		guard data.count == expectedBytes else {
			throw BenchInputError.sizeMismatch("\(filename).bin: expected \(expectedBytes) bytes, got \(data.count)")
		}
		data.withUnsafeBytes { raw in
			memcpy(ptr, raw.baseAddress!, expectedBytes)
		}
		return mel
	}

	/// Mel .bin files land alongside everything else in the bundle root
	/// (see `makeInput`), so this only picks up top-level `.bin` entries —
	/// the compiled models' internal `weight.bin`/`coremldata.bin` sit one
	/// level down inside their `.mlmodelc` directories and aren't listed here.
	private static func firstBundledMelFilename() -> String? {
		guard let resourceURL = Bundle.main.resourceURL else { return nil }
		let files = (try? FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil)) ?? []
		return files
			.filter { $0.pathExtension == "bin" }
			.map { $0.deletingPathExtension().lastPathComponent }
			.sorted()
			.first
	}

	/// Packed windows (from convert/export_mel_bin_packed.py) also land in the
	/// bundle root, named `window000.bin`, `window001.bin`, ... — sorted
	/// lexicographically that's also numeric order since the index is
	/// zero-padded.
	private static func bundledPackedWindowFilenames() -> [String] {
		guard let resourceURL = Bundle.main.resourceURL else { return [] }
		let files = (try? FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil)) ?? []
		return files
			.filter { $0.pathExtension == "bin" && $0.deletingPathExtension().lastPathComponent.hasPrefix("window") }
			.map { $0.deletingPathExtension().lastPathComponent }
			.sorted()
	}

	private func timeOneInference(_ s: Session) throws -> Double {
		let t = CACurrentMediaTime()
		try s.predict()
		return (CACurrentMediaTime() - t) * 1000
	}
}
