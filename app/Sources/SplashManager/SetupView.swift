import SwiftUI
import AppKit
import SplashManagerCore

/// First-run help: what this Mac offers, whether Splash is installed, how to install it,
/// and that no model is downloaded without a confirmation.
struct SetupView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var supervisor: SplashSupervisor
    @EnvironmentObject var store: ConfigStore
    @State private var installing = false
    @State private var installError: String?
    private let checks = SystemRequirements.current()

    var needed: Bool {
        supervisor.install == nil || store.configs.isEmpty || checks.contains { $0.ok != true }
    }

    var body: some View {
        if needed {
            GroupBox("Set up") {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(checks) { check in
                        Label {
                            VStack(alignment: .leading) {
                                Text(check.title)
                                Text(check.detail).font(.caption).foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: check.ok == true ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(check.ok == true ? .green : .red)
                        }
                    }
                    if checks.contains(where: { $0.ok != true }) {
                        Text("Splash will most likely refuse to start on this Mac. Its own device check decides at the start.")
                            .font(.callout).foregroundStyle(.orange)
                    }
                    Divider()
                    splashStep
                    Divider()
                    modelStep
                }.padding(6)
            }
        }
    }

    @ViewBuilder private var splashStep: some View {
        if let install = supervisor.install {
            Label("Splash \(install.version) found at \(install.executable.path)", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        } else {
            Label("Splash is not installed", systemImage: "xmark.circle.fill").foregroundStyle(.red)
            if let brew = Homebrew.find() {
                Text("This installs the Splash engine with Homebrew (`brew install \(Homebrew.tap)`, about 235 MB). It downloads no model.")
                    .font(.callout).foregroundStyle(.secondary)
                HStack {
                    Button(installing ? "Installing…" : "Install Splash with Homebrew") { install(brew) }
                        .disabled(installing)
                    if installing { ProgressView().controlSize(.small) }
                }
                if let installError { Text(installError).foregroundStyle(.red).font(.callout) }
            } else {
                Text("Homebrew is not installed. Install it from brew.sh (it asks for your password and may install the Xcode command line tools), then run:")
                    .font(.callout).foregroundStyle(.secondary)
                CopyField(label: "Command", value: "brew install \(Homebrew.tap)")
                Link("Open brew.sh", destination: URL(string: "https://brew.sh")!)
                Text("This app does not install Homebrew for you.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var modelStep: some View {
        if store.configs.isEmpty {
            Label("No model is configured", systemImage: "circle").foregroundStyle(.secondary)
            let suggestions = supervisor.install.map { SplashFamilies.suggestedModels(executable: $0.executable) } ?? []
            Text("Pick a model that your installed Splash suggests. Adding it downloads nothing; the first start shows the size and asks first.")
                .font(.callout).foregroundStyle(.secondary)
            ForEach(suggestions, id: \.self) { id in
                Button("Add \(id)") {
                    let name = id.split(separator: "/").last.map(String.init) ?? id
                    let config = ModelConfig(id: ModelConfig.makeId(modelId: id, revision: nil, existing: Set(store.configs.map(\.id))), displayName: name, modelId: id)
                    if let saved = try? store.save(config), store.settings.selectedConfigId == nil { try? store.updateSettings { $0.selectedConfigId = saved.id } }
                }
            }
            Button("Import models Splash already installed") { env.importDetectedModels() }
        } else {
            Label("\(store.configs.count) model configuration(s)", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        }
    }

    private func install(_ brew: URL) {
        guard Dialogs.confirm("Install Splash?", "This runs `brew install \(Homebrew.tap)` and downloads about 235 MB. No model is downloaded.", ok: "Install") else { return }
        installing = true; installError = nil
        let logs = env.logs
        Task {
            let status = await Homebrew.installSplash(brew: brew) { line in Task { @MainActor in logs.append(line, source: .app) } }
            installing = false
            if status == 0 { await supervisor.refreshInstall() } else { installError = "brew exited with status \(status). See the Logs tab." }
        }
    }
}
