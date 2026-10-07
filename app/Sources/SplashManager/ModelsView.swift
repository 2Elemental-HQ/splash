import SwiftUI
import SplashManagerCore

struct ModelsView: View {
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var supervisor: SplashSupervisor
    @EnvironmentObject var store: ConfigStore
    @State private var editing: ModelConfig?
    @State private var isNew = false
    @State private var estimates: [String: String] = [:]

    var body: some View {
        let views = supervisor.configViews()
        VStack(alignment: .leading) {
            HStack {
                Text("One model is loaded per Splash instance. Switching stops the running model and starts the other, and is refused while calls run.")
                    .font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Import installed") { env.importDetectedModels() }
                    .help("Add models that Splash installed, with their pinned revision")
                Button("Add model…") {
                    editing = ModelConfig(id: "", displayName: "", modelId: ""); isNew = true
                }.buttonStyle(.borderedProminent)
            }.padding([.horizontal, .top], 16)
            List {
                ForEach(views, id: \.id) { view in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(view.displayName).font(.headline)
                            Pill(text: view.availability.title, color: view.availability.color)
                            if view.active { Pill(text: "active", color: .green) }
                            if view.verifiedStartAt != nil { Pill(text: "start verified", color: .green) }
                            if view.selected { Pill(text: "selected", color: .accentColor) }
                            Spacer()
                            Button("Edit") { editing = store.config(id: view.id); isNew = false }
                            Button("Select") { try? store.updateSettings { $0.selectedConfigId = view.id } }.disabled(view.selected)
                            Button("Remove", role: .destructive) { try? store.remove(id: view.id) }.disabled(view.active)
                        }
                        Text(view.modelId + (view.revision.map { "  @ \($0)" } ?? "  (default branch)"))
                            .font(.system(.callout, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                        Text("id: \(view.id)").font(.caption).foregroundStyle(.tertiary)
                        if view.downloadRequired {
                            HStack {
                                Text("Download needed: \(view.missing.joined(separator: ", "))").font(.callout).foregroundStyle(.orange)
                                Button(estimates[view.id] == nil ? "Check size" : estimates[view.id]!) {
                                    estimates[view.id] = "Checking…"
                                    Task {
                                        let list = await DownloadEstimator.estimate(modelId: view.modelId, revision: view.revision, missing: view.missing)
                                        let known = list.compactMap(\.bytes).reduce(0, +)
                                        let unknown = list.contains { $0.bytes == nil } || view.missing.contains("(draft unknown)")
                                        estimates[view.id] = (known > 0 ? "About \(formatBytes(known))" : "Size unknown") + (unknown && known > 0 ? " or more" : "")
                                    }
                                }.buttonStyle(.link)
                            }
                        }
                        if let note = view.note { Text(note).font(.caption).foregroundStyle(.secondary) }
                    }.padding(.vertical, 4)
                }
            }
        }
        .navigationTitle("Models")
        .sheet(item: $editing) { config in
            ModelEditor(config: config, isNew: isNew) { editing = nil }
                .environmentObject(store).environmentObject(supervisor)
        }
    }
}

struct ModelEditor: View {
    @EnvironmentObject var store: ConfigStore
    @EnvironmentObject var supervisor: SplashSupervisor
    @State var config: ModelConfig
    let isNew: Bool
    let done: () -> Void
    @State private var error: String?

    private var install: SplashInstall? { supervisor.install }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(isNew ? "Add model" : "Edit model").font(.title2)
            Form {
                TextField("Name", text: $config.displayName)
                TextField("Model ID", text: $config.modelId, prompt: Text("owner/repo or owner/repo:VARIANT"))
                    .disabled(!isNew && supervisor.activeConfig?.id == config.id)
                TextField("Revision", text: Binding(get: { config.revision ?? "" }, set: { config.revision = $0.isEmpty ? nil : $0 }),
                          prompt: Text("optional: commit, tag or branch"))
                Section("Options the installed Splash offers") {
                    if install == nil { Text("Splash was not found, so no options are listed.").foregroundStyle(.secondary) }
                    if install?.supports("--language-only") == true { Toggle("Text only (skip vision)", isOn: $config.options.languageOnly) }
                    if install?.supports("--max-context") == true {
                        TextField("Max context", text: Binding(get: { config.options.maxContext ?? "" }, set: { config.options.maxContext = $0.isEmpty ? nil : $0 }), prompt: Text("e.g. 100K"))
                    }
                    if install?.supports("--max-memory") == true {
                        TextField("Max memory", text: Binding(get: { config.options.maxMemory ?? "" }, set: { config.options.maxMemory = $0.isEmpty ? nil : $0 }), prompt: Text("e.g. 28G"))
                    }
                    if install?.supports("--idle-release") == true {
                        TextField("Idle release", text: Binding(get: { config.options.idleRelease ?? "" }, set: { config.options.idleRelease = $0.isEmpty ? nil : $0 }), prompt: Text("e.g. 10m or off"))
                    }
                    if install?.supports("--default-reasoning-effort") == true {
                        Picker("Default thinking", selection: $config.options.reasoning) {
                            Text("Model default").tag(ReasoningDefault.modelDefault)
                            Text("Off (effort none)").tag(ReasoningDefault.off)
                        }
                        Text("Only values with a verified meaning are offered. Splash 1.3.0 turns thinking off for `none` on the tested Qwen model; other levels depend on the model's template.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if install?.supports("--disable-ane") == true { Toggle("Prefill on the GPU only (no Neural Engine)", isOn: $config.options.disableANE) }
                }
            }.formStyle(.grouped)
            if let error { Text(error).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
                Button("Save") { save() }.keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
            }
        }
        .padding(20).frame(width: 560, height: 560)
    }

    private func save() {
        do {
            var c = config
            if isNew {
                c.id = ModelConfig.makeId(modelId: c.modelId, revision: c.revision, existing: Set(store.configs.map(\.id)))
                if c.displayName.trimmingCharacters(in: .whitespaces).isEmpty { c.displayName = c.modelId.split(separator: "/").last.map(String.init) ?? c.modelId }
            }
            let saved = try store.save(c)
            if store.settings.selectedConfigId == nil { try store.updateSettings { $0.selectedConfigId = saved.id } }
            done()
        } catch let failure as Validation.Failure { error = failure.description }
        catch { self.error = "\(error)" }
    }
}
