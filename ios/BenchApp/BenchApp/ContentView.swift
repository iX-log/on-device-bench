import SwiftUI

struct ContentView: View {
	@State private var vm = BenchmarkViewModel()

	var body: some View {
		ScrollView {
			VStack(alignment: .leading, spacing: 12) {
				Text(vm.log).monospaced().font(.caption)
				if !vm.progress.isEmpty {
					Text(vm.progress).monospaced().font(.caption).foregroundStyle(.orange)
				}
			}
			.frame(maxWidth: .infinity, alignment: .leading)
			.padding()
		}
		.safeAreaInset(edge: .bottom) {
			VStack(spacing: 12) {
				Text(vm.selectedModelInfo).monospaced().font(.caption).foregroundStyle(.secondary)

				Picker("Precision", selection: $vm.selected) {
					ForEach(Precision.allCases) { p in
						Text(p.label).tag(p)
					}
				}
				.pickerStyle(.segmented)
				.disabled(vm.isRunning)

				Toggle("Real input (LibriSpeech mel)", isOn: $vm.useRealInput)
					.disabled(vm.isRunning)

				HStack(spacing: 16) {
					Button("Quick (100)") { Task { await vm.runQuick() } }
					Button("Sustained (10 min)") { Task { await vm.runSustained() } }
					Button("Dump features") { Task { await vm.dumpFeatures() } }
				}
				.disabled(vm.isRunning)
			}
			.padding()
		}
	}
}
