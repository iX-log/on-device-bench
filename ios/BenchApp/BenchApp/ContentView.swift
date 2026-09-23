import SwiftUI

struct ContentView: View {
	@State private var vm = BenchmarkViewModel()
	@State private var showThermalConfirm = false

	var body: some View {
		ScrollView {
			VStack(alignment: .leading, spacing: 12) {
				if !vm.ceilingStatus.isEmpty {
					Text(vm.ceilingStatus).monospaced().font(.caption).foregroundStyle(.secondary)
				}
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
					Button("Sustained (10 min)") {
						if vm.thermalWarning != nil {
							showThermalConfirm = true
						} else {
							Task { await vm.runSustained() }
						}
					}
					Button("Dump features") { Task { await vm.dumpFeatures() } }
				}
				.disabled(vm.isRunning)
				.confirmationDialog(
					vm.thermalWarning ?? "",
					isPresented: $showThermalConfirm,
					titleVisibility: .visible
				) {
					Button("Run anyway", role: .destructive) { Task { await vm.runSustained() } }
					Button("Cancel", role: .cancel) {}
				}

				HStack(spacing: 16) {
					Button("Memory ceiling") { Task { await vm.runMemoryCeiling() } }
						.disabled(vm.isRunning)
					Button("Clear ceiling data", role: .destructive) { vm.clearCeilingProgress() }
						.disabled(vm.isRunning)
				}
			}
			.padding()
		}
	}
}
