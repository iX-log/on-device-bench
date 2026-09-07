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
			HStack(spacing: 16) {
				Button("Quick (100)") { Task { await vm.runQuick() } }
				Button("Sustained (10 min)") { Task { await vm.runSustained() } }
			}
			.disabled(vm.isRunning)
			.padding()
		}
	}
}
