import SwiftUI
import AppKit
import ServiceManagement
import SplashManagerCore

struct SettingsView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var store: ConfigStore
    @EnvironmentObject var supervisor: SplashSupervisor
    @EnvironmentObject var service: ManagementService
    @State private var error: String?
    @State private var allowedHosts = ""
    @State private var launchAtLogin = SMAppService.mainApp.status == .enabled
    @State private var inferencePort = ""
    @State private var managementPort = ""

    var body: some View {
        let s = store.settings
        Form {
            Section("Splash") {
                LabeledContent("Executable") {
                    Text(supervisor.install?.executable.path ?? "not found").textSelection(.enabled)
                }
                LabeledContent("Version", value: supervisor.install?.version ?? "unknown")
                HStack {
                    TextField("Path override", text: Binding(get: { s.splashPath ?? "" }, set: { v in apply { $0.splashPath = v.isEmpty ? nil : v } }), prompt: Text("default: Homebrew"))
                    Button("Choose…") { choose() }
                }
            }
            Section("Inference endpoint") {
                TextField("Port", text: $inferencePort).onSubmit { if let p = Int(inferencePort) { apply { $0.inferencePort = p } } }
                Picker("Reachable from", selection: Binding(get: { s.inferenceExposure }, set: { v in apply { $0.inferenceExposure = v } })) {
                    Text("This Mac only (127.0.0.1)").tag(Exposure.loopback)
                    Text("All network interfaces + API key (reach it over Tailscale)").tag(Exposure.allInterfaces)
                }
                if s.inferenceExposure != .loopback {
                    Text("Splash then listens on every interface, the LAN included, and requires the API key shown on Overview. Restrict the LAN with the macOS firewall if needed. \(NetInfo.tailnetAddress().map { "Tailscale address: \($0)." } ?? "No Tailscale address found.")")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        TextField("Allowed host names", text: $allowedHosts, prompt: Text("mymac.tailnet.ts.net, comma separated"))
                            .onSubmit { saveHosts() }
                        Button("Apply") { saveHosts() }
                    }
                    Button("Create a new API key") { env.regenerate(SecretAccount.inferenceKey) }
                }
                Text("Changes apply at the next start.").font(.caption).foregroundStyle(.secondary)
            }
            Section("Management API (for AgentOS)") {
                Toggle("Enabled", isOn: Binding(get: { s.managementEnabled }, set: { v in apply { $0.managementEnabled = v } }))
                TextField("Port", text: $managementPort).onSubmit { if let p = Int(managementPort) { apply { $0.managementPort = p } } }
                Picker("Reachable from", selection: Binding(get: { s.managementExposure }, set: { v in apply { $0.managementExposure = v } })) {
                    Text("This Mac only (127.0.0.1)").tag(Exposure.loopback)
                    Text("Tailscale address and this Mac").tag(Exposure.tailnet)
                }
                LabeledContent("Listening", value: service.listening.isEmpty ? "off" : service.listening.joined(separator: ", "))
                if let problem = service.problem { Text(problem).foregroundStyle(.orange).font(.caption) }
                CopyField(label: "Token", value: env.managementToken(), secret: true)
                Button("Create a new token") { env.regenerate(SecretAccount.managementToken) }
                Text("The API runs inside this app, so it stays reachable while Splash is stopped and while this window is closed. It needs the app to run: turn on “Open at login”.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Behaviour") {
                Toggle("Open at login", isOn: $launchAtLogin).onChange(of: launchAtLogin) { _, on in
                    do { try AppEnvironment.setLaunchAtLogin(on) } catch { self.error = "Login item: \(error.localizedDescription)"; launchAtLogin = SMAppService.mainApp.status == .enabled }
                }
                Toggle("Start Splash when this app opens", isOn: Binding(get: { s.startSplashWhenAppLaunches }, set: { v in apply { $0.startSplashWhenAppLaunches = v } }))
                Toggle("Stop Splash when this app quits", isOn: Binding(get: { s.stopSplashWhenAppQuits }, set: { v in apply { $0.stopSplashWhenAppQuits = v } }))
                Toggle("Restart after a crash (at most 3 times)", isOn: Binding(get: { s.autoRestart }, set: { v in apply { $0.autoRestart = v } }))
                Toggle("Save redacted logs to a file", isOn: Binding(get: { s.persistLogsToFile }, set: { v in apply { $0.persistLogsToFile = v } }))
            }
            if let error { Text(error).foregroundStyle(.red) }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear {
            inferencePort = String(s.inferencePort); managementPort = String(s.managementPort)
            allowedHosts = s.allowedHosts.joined(separator: ", ")
            Task { await supervisor.refreshInstall() }
        }
    }

    private func apply(_ change: (inout ManagerSettings) -> Void) {
        do {
            try store.updateSettings(change)
            error = nil
            supervisor.settingsChanged()
            service.reconcile()
        } catch let failure as Validation.Failure { error = failure.description }
        catch { self.error = "\(error)" }
    }

    private func saveHosts() {
        let list = allowedHosts.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        apply { $0.allowedHosts = list }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.directoryURL = URL(fileURLWithPath: "/opt/homebrew/bin")
        if panel.runModal() == .OK, let url = panel.url { apply { $0.splashPath = url.path } }
    }
}
