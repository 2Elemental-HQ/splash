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
    @State private var showAdvanced = false
    @State private var copied = false

    var body: some View {
        let s = store.settings
        Form {
            Section("General") {
                Toggle("Open Splash Manager at login", isOn: $launchAtLogin).onChange(of: launchAtLogin) { _, on in
                    do { try AppEnvironment.setLaunchAtLogin(on) } catch { self.error = "Login item: \(error.localizedDescription)"; launchAtLogin = SMAppService.mainApp.status == .enabled }
                }
                Toggle("Start Splash when Splash Manager opens", isOn: Binding(get: { s.startSplashWhenAppLaunches }, set: { v in apply { $0.startSplashWhenAppLaunches = v } }))
                Toggle("Stop Splash when Splash Manager quits", isOn: Binding(get: { s.stopSplashWhenAppQuits }, set: { v in apply { $0.stopSplashWhenAppQuits = v } }))
                Toggle("Restart Splash after a crash (at most 3 times)", isOn: Binding(get: { s.autoRestart }, set: { v in apply { $0.autoRestart = v } }))
            }

            Section("Access") {
                Picker("Splash is reachable from", selection: Binding(get: { s.inferenceExposure == .loopback ? Exposure.loopback : Exposure.allInterfaces }, set: { v in apply { $0.inferenceExposure = v } })) {
                    Text("This Mac only").tag(Exposure.loopback)
                    Text("Other devices on the network (needs an API key)").tag(Exposure.allInterfaces)
                }
                if s.inferenceExposure != .loopback {
                    Text("Splash then listens on every network interface, the local network included, and requires the API key below from every client. Restrict the local network with the macOS firewall if needed.")
                        .font(.caption).foregroundStyle(.secondary)
                    CopyField(label: "API key", value: env.secrets.read(SecretAccount.inferenceKey) ?? "(created at the next start)", secret: true)
                    Button("Create a new API key") { env.regenerate(SecretAccount.inferenceKey) }
                }
                Text("Changes to access apply the next time Splash starts. Until then the Overview shows what the running Splash really uses.")
                    .font(.caption).foregroundStyle(.secondary)

                Toggle("Allow remote control of Splash Manager", isOn: Binding(get: { s.managementEnabled }, set: { v in apply { $0.managementEnabled = v } }))
                if s.managementEnabled {
                    Picker("Remote control is reachable from", selection: Binding(get: { s.managementExposure }, set: { v in apply { $0.managementExposure = v } })) {
                        Text("This Mac only").tag(Exposure.loopback)
                        Text("This Mac and my Tailscale network").tag(Exposure.tailnet)
                    }
                    CopyField(label: "Access token", value: env.managementToken(), secret: true)
                    Button("Create a new access token") { env.regenerate(SecretAccount.managementToken) }
                    Text("Other programs and devices use this token to start, stop and switch Splash through a local HTTP API. It works while Splash is stopped and while this window is closed, as long as Splash Manager runs. Open it at login to have it always.")
                        .font(.caption).foregroundStyle(.secondary)
                    LabeledContent("Listening", value: service.listening.isEmpty ? "off" : service.listening.joined(separator: ", "))
                    if let problem = service.problem { Text(problem).foregroundStyle(.orange).font(.caption) }
                }
            }

            Section("Splash") {
                LabeledContent("Version", value: supervisor.install?.version ?? "not found")
                LabeledContent("Runtime", value: runtimeDescription)
                if let problem = supervisor.runtimeProblem { Text(problem).foregroundStyle(.red).font(.caption) }
            }

            Section {
                DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Technical options. The defaults are right for most people.").font(.caption).foregroundStyle(.secondary)
                        HStack {
                            TextField("Splash executable", text: Binding(get: { s.splashPath ?? "" }, set: { v in apply { $0.splashPath = v.isEmpty ? nil : v } }),
                                      prompt: Text("default: the runtime bundled with this app"))
                            Button("Choose…") { choose() }
                            if s.splashPath != nil { Button("Use default") { apply { $0.splashPath = nil } } }
                        }
                        Toggle("Prefer a Splash installed on this Mac over the bundled one", isOn: Binding(get: { s.preferInstalledSplash }, set: { v in apply { $0.preferInstalledSplash = v } }))
                        Text("This app never changes an installed Splash. Stop and switch wait for running calls only when the running Splash reports that it can; the Overview says which.")
                            .font(.caption).foregroundStyle(.secondary)
                        TextField("Splash port", text: $inferencePort).onSubmit { if let p = Int(inferencePort) { apply { $0.inferencePort = p } } }
                        TextField("Remote control port", text: $managementPort).onSubmit { if let p = Int(managementPort) { apply { $0.managementPort = p } } }
                        HStack {
                            TextField("Allowed host names", text: $allowedHosts, prompt: Text("extra names clients use, comma separated"))
                                .onSubmit { saveHosts() }
                            Button("Apply") { saveHosts() }
                        }
                        Text("Needed when other devices reach Splash by a name. This Mac's Tailscale address and name are added on their own.").font(.caption).foregroundStyle(.secondary)
                        Toggle("Serve Splash's chat page", isOn: Binding(get: { s.serveWebChat }, set: { v in apply { $0.serveWebChat = v } }))
                        Toggle("Save redacted logs to a file", isOn: Binding(get: { s.persistLogsToFile }, set: { v in apply { $0.persistLogsToFile = v } }))
                        HStack {
                            Button(copied ? "Copied" : "Copy diagnostics") { copyDiagnostics() }
                            Button("Show state folder") { NSWorkspace.shared.activateFileViewerSelecting([store.paths.support]) }
                        }
                    }.padding(.top, 6)
                }
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

    private var runtimeDescription: String {
        switch supervisor.install?.source {
        case .bundled?: return "Bundled with this app, integrity verified"
        case .installed?: return "Installed on this Mac (not changed by this app)"
        case .custom?: return "Custom executable"
        case nil: return "none"
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
        if panel.runModal() == .OK, let url = panel.url { apply { $0.splashPath = url.path } }
    }

    /// The status the API would give, as text. It holds no secret.
    private func copyDiagnostics() {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let text = (try? encoder.encode(supervisor.snapshot())).map { String(decoding: $0, as: UTF8.self) } ?? "unavailable"
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text + "\n\n" + env.logs.exportText, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
    }
}
