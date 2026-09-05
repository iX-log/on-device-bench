import Playgrounds
import SwiftUI
import CoreML

struct ContentView: View {
	@State private var log = "ready"

	var body: some View {
		ScrollView {
			Text(log).monospaced().font(.caption).padding()
		}
		.safeAreaInset(edge: .bottom) {
			Button("Run") { Task { await run() } }.padding()
		}
	}

	func run() async {
		var out = ""
		do {
			let config = MLModelConfiguration()
			config.computeUnits = .all

			let t0 = CACurrentMediaTime()
			let model = try whisper_base_encoder_fp16(configuration: config)
			let loadTime = CACurrentMediaTime() - t0
			out += String(format: "load: %.1f ms\n", loadTime * 1000)

			let mel = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
			let ptr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
			for i in 0..<mel.count { ptr[i] = Float.random(in: -1...1) }

			for i in 1...10 {
				let t = CACurrentMediaTime()
				_ = try model.prediction(mel: mel)
				let ms = (CACurrentMediaTime() - t) * 1000
				out += String(format: "run %2d: %6.1f ms\n", i, ms)
			}
			log = out
		} catch {
			log = "error: \(error)"
		}
	}
}
#Preview {
    ContentView()
}

#Playground {
    _ = 1 + 2
}
