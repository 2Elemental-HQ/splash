import SwiftUI
import AppKit
import SplashManagerCore

enum Tab: String, CaseIterable, Identifiable {
    case overview = "Overview", models = "Models", settings = "Settings", logs = "Logs"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .overview: "gauge.with.dots.needle.50percent"
        case .models: "cpu"
        case .settings: "gearshape"
        case .logs: "text.alignleft"
        }
    }
}

struct MainWindow: View {
    @State private var tab: Tab? = .overview
    var body: some View {
        NavigationSplitView {
            List(Tab.allCases, selection: $tab) { tab in
                Label(tab.rawValue, systemImage: tab.symbol).tag(tab)
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 170)
        } detail: {
            switch tab ?? .overview {
            case .overview: OverviewView()
            case .models: ModelsView()
            case .settings: SettingsView()
            case .logs: LogsView()
            }
        }
    }
}

struct OverviewView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var supervisor: SplashSupervisor
    @EnvironmentObject var store: ConfigStore

    var body: some View {
        let status = supervisor.snapshot()
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    StatusBadge(state: status.state, external: status.ownership == .external)
                    if let detail = status.detail { Text(detail).foregroundStyle(.secondary).lineLimit(2) }
                    Spacer()
                }
                if let conflict = status.conflict {
                    Label(conflict.message, systemImage: "exclamationmark.circle")
                        .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
                }
                if let error = status.lastError, status.state == .failed || status.state == .stopped {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Last error: \(error.code)", systemImage: "xmark.octagon").foregroundStyle(.red)
                        Text(error.message).textSelection(.enabled)
                        if let retry = status.retry {
                            Text("Restart \(retry.attempt) of \(retry.maxAttempts) is scheduled.").foregroundStyle(.secondary)
                        }
                    }
                    .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                }

                GroupBox("Model") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Picker("Model", selection: Binding(
                                get: { store.settings.selectedConfigId ?? "" },
                                set: { id in try? store.updateSettings { $0.selectedConfigId = id } })) {
                                ForEach(store.configs) { Text($0.displayName).tag($0.id) }
                            }
                            .frame(maxWidth: 360)
                            .disabled(store.configs.isEmpty)
                            controls(status)
                        }
                        let selected = store.settings.selectedConfigId.flatMap { store.config(id: $0) }
                        if let selected {
                            let assessment = supervisor.assess(selected)
                            HStack {
                                Text(selected.modelId).font(.system(.body, design: .monospaced))
                                if let rev = selected.revision { Pill(text: "rev \(rev.prefix(8))", color: .secondary) }
                                Pill(text: assessment.availability.title, color: assessment.availability.color)
                            }
                            if assessment.availability == .notLocal {
                                Text("Starting needs a download: \(assessment.missing.joined(separator: ", ")). You confirm it first.")
                                    .font(.callout).foregroundStyle(.orange)
                            }
                        } else {
                            Text("Add a model in the Models tab.").foregroundStyle(.secondary)
                        }
                        Divider()
                        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 6) {
                            GridRow { Text("Loaded model").foregroundStyle(.secondary); Text(supervisor.loadedModelId ?? "none") }
                            GridRow { Text("Splash version").foregroundStyle(.secondary); Text(status.splash.version ?? "not found") }
                            GridRow {
                                Text("Readiness").foregroundStyle(.secondary)
                                HStack(spacing: 10) {
                                    check("process", status.readiness.processAlive)
                                    check("HTTP ready", status.readiness.httpReady)
                                    check("model loaded", status.readiness.modelLoaded)
                                }
                            }
                            GridRow {
                                Text("Activity").foregroundStyle(.secondary)
                                Text(status.activity.activeRequests.map { "\($0) active request(s)" } ?? "unknown")
                            }
                            if let process = status.process {
                                GridRow {
                                    Text("Process").foregroundStyle(.secondary)
                                    Text("pid \(process.pid), up \(process.uptimeSeconds) s\(process.adopted ? ", adopted after app restart" : "")")
                                }
                            }
                        }
                    }.padding(6)
                }

                GroupBox("Endpoint") {
                    VStack(alignment: .leading, spacing: 8) {
                        CopyField(label: "Local", value: status.endpoint.localUrl + "/v1")
                        if status.endpoint.exposure != .loopback, let url = status.endpoint.tailnetUrl {
                            CopyField(label: "Tailnet", value: url + "/v1")
                        }
                        Text(status.endpoint.exposure == .loopback
                             ? "Bound to 127.0.0.1 only. Other devices cannot reach it."
                             : "Bound to \(status.endpoint.bindHost). An API key is required.")
                            .font(.callout).foregroundStyle(.secondary)
                        if status.endpoint.requiresApiKey {
                            CopyField(label: "API key", value: env.secrets.read(SecretAccount.inferenceKey) ?? "(created at the next start)", secret: true)
                        }
                    }.padding(6)
                }
            }
            .padding(20)
        }
        .navigationTitle("Overview")
    }

    @ViewBuilder
    private func controls(_ status: ManagerStatus) -> some View {
        switch (status.ownership, status.state) {
        case (.external, _):
            Text("Not managed here").foregroundStyle(.secondary)
        case (_, .stopping):
            ProgressView().controlSize(.small)
        case (.managed, .ready), (.managed, .starting):
            Button("Stop") { Task { await StopFlow.run(supervisor) } }
            if let id = store.settings.selectedConfigId, id != status.config?.id {
                Button("Switch to selected") { Task { await SwitchFlow.run(supervisor, configId: id) } }
            }
        default:
            Button("Start") { Task { await StartFlow.run(supervisor, configId: store.settings.selectedConfigId) } }
                .buttonStyle(.borderedProminent)
                .disabled(store.settings.selectedConfigId == nil)
        }
    }

    private func check(_ title: String, _ ok: Bool) -> some View {
        Label(title, systemImage: ok ? "checkmark.circle.fill" : "circle").foregroundStyle(ok ? .green : .secondary)
    }
}

struct LogsView: View {
    @EnvironmentObject var logs: LogBuffer
    @EnvironmentObject var store: ConfigStore
    @EnvironmentObject var supervisor: SplashSupervisor

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(logs.lines) { line in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(line.date, format: .dateTime.hour().minute().second()).foregroundStyle(.secondary)
                                Text(line.source == .app ? "app" : "splash").foregroundStyle(line.source == .app ? .blue : .green).frame(width: 44, alignment: .leading)
                                Text(line.text).textSelection(.enabled)
                            }
                            .font(.system(.caption, design: .monospaced)).id(line.id)
                        }
                    }.padding(10)
                }
                .onChange(of: logs.lines.last?.id) { _, id in if let id { proxy.scrollTo(id, anchor: .bottom) } }
            }
            Divider()
            HStack {
                Text(store.settings.persistLogsToFile
                     ? "Saving redacted logs to ~/Library/Logs/Splash Manager."
                     : "Logs stay in memory. Prompts, answers and credentials are never logged.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(logs.exportText, forType: .string) }
                Button("Clear") { logs.clear() }
                Button("Reveal log folder") { NSWorkspace.shared.open(store.paths.logsDirectory.deletingLastPathComponent().appendingPathComponent("Splash Manager")) }
                    .disabled(!store.settings.persistLogsToFile)
            }.padding(8)
        }
        .navigationTitle("Logs")
    }
}
