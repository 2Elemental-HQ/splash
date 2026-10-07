import Foundation

public struct SupervisorTiming: Sendable {
    public var tick: TimeInterval = 1.0
    public var startTimeoutLocal: TimeInterval = 300
    public var startTimeoutDownload: TimeInterval = 3600
    public var termGrace: TimeInterval = 40
    public var intGrace: TimeInterval = 10
    public var notReadyTicksBeforeStarting = 4
    public var retryDelays: [TimeInterval] = [5, 20, 60]
    public var stableAfter: TimeInterval = 600
    public var idleRecheck: TimeInterval = 0.4
    /// A draining Splash with no call left that has not exited after this long is stopped by signal.
    /// A running call is never the reason for this timeout.
    public var drainIdleEscalation: TimeInterval = 60
    public var drainPoll: TimeInterval = 0.1
    public init() {}
}

/// Owns the one `splash serve` process of this app.
///
/// Everything runs on the main actor, but an `await` still lets other calls in.
/// So every lifecycle change (start, stop, switch, restart, adopt) first claims
/// the single `op` slot, before its first suspension point, and checks after
/// each suspension that it still holds it. The process it acts on is checked
/// too, by pid and kernel start time.
@MainActor
public final class SplashSupervisor: ObservableObject {
    // MARK: Published state
    @Published public private(set) var state: RunState = .stopped
    @Published public private(set) var ownership: Ownership = .none
    @Published public private(set) var detail: String?
    /// The configuration of the running process, as it was applied. nil when nothing runs.
    @Published public private(set) var applied: AppliedConfig?
    @Published public private(set) var loadedModelId: String?
    @Published public private(set) var httpReady = false
    @Published public private(set) var modelLoaded = false
    @Published public private(set) var activity: SplashClient.Activity?
    @Published public private(set) var lastError: ErrorRecord?
    @Published public private(set) var retry: ManagerStatus.Retry?
    @Published public private(set) var conflict: ManagerStatus.Conflict?
    @Published public private(set) var install: SplashInstall?
    /// Why a Splash that exists is not used, for example a bundled runtime that fails its check.
    @Published public private(set) var runtimeProblem: String?
    @Published public private(set) var startedAt: Date?
    @Published public private(set) var pid: Int32?
    @Published public private(set) var adopted = false
    @Published public private(set) var tailnetAddress: String?
    /// Whether the running Splash can drain: read from its status, never assumed.
    @Published public private(set) var drainSupported: Bool?
    @Published public private(set) var draining = false
    @Published public private(set) var operationKind: String?

    public let store: ConfigStore
    public let logs: LogBuffer
    public let startedAtApp = Date()
    public var timing: SupervisorTiming
    public var cache = HFCache()
    /// Tests give a family table; otherwise the installed Splash's own table is used.
    public var familiesOverride: [SplashFamily]?
    /// Test hook: runs after a stop is accepted and before its signal is sent.
    public var beforeStopSignal: (@MainActor () async -> Void)?
    public static let managerVersion = "0.2.0"

    private enum OpKind: String { case start, stop, switchTo = "switch", retry, adopt }
    private struct Operation {
        let id = UUID()
        let kind: OpKind
        var spec: EffectiveSpec?
    }

    private let secrets: SecretStore
    private let paths: AppPaths
    private let tail: FileTail
    private var identity: ProcessIdentity?
    private var runKey: String?
    private var readyOnce = false
    private var readyAt: Date?
    private var startDeadline = Date()
    private var failedProbes = 0
    private var failureCount = 0
    private var desiredRunning = false
    private var ticking = false
    private var op: Operation?
    private var retryTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var opTask: Task<Void, Never>?
    private var installInspectedFor: URL?
    private var assessCache: [String: (Date, HFCache.Assessment)] = [:]

    public init(store: ConfigStore, secrets: SecretStore, logs: LogBuffer, timing: SupervisorTiming = SupervisorTiming()) {
        self.store = store
        self.secrets = secrets
        self.logs = logs
        self.timing = timing
        self.paths = store.paths
        self.tail = FileTail(url: store.paths.consoleFile)
        logs.persistTo = store.settings.persistLogsToFile ? store.paths.persistentLog : nil
    }

    // MARK: Lifecycle of the supervisor

    public func begin() {
        guard pollTask == nil else { return }
        let adoptToken = try? claim(.adopt, spec: nil)
        pollTask = Task { [weak self] in
            guard let self else { return }
            await self.refreshInstall()
            await self.adoptIfPossible()
            if let adoptToken { self.release(adoptToken) }
            if self.store.settings.startSplashWhenAppLaunches, self.state == .stopped, self.ownership == .none,
               let id = self.store.settings.selectedConfigId {
                do { _ = try await self.start(configId: id, allowDownload: false) }
                catch let error as AppError { self.log("Auto start refused: \(error.message)") }
                catch { self.log("Auto start failed: \(error)") }
            }
            while !Task.isCancelled {
                await self.tick()
                try? await Task.sleep(nanoseconds: UInt64(self.timing.tick * 1_000_000_000))
            }
        }
    }

    public func shutdownSupervisor() { pollTask?.cancel(); pollTask = nil; retryTask?.cancel(); opTask?.cancel() }

    // MARK: Operation slot

    private func claim(_ kind: OpKind, spec: EffectiveSpec?) throws -> UUID {
        if let current = op {
            throw AppError("busy", "Another operation is in progress: \(current.kind.rawValue).",
                           details: ["operation": current.kind.rawValue])
        }
        let operation = Operation(kind: kind, spec: spec)
        op = operation
        operationKind = kind.rawValue
        return operation.id
    }

    /// Takes the slot from whatever holds it. Only a person's explicit force uses this.
    private func takeOver(_ kind: OpKind, spec: EffectiveSpec?) -> UUID {
        opTask?.cancel(); opTask = nil
        let operation = Operation(kind: kind, spec: spec)
        op = operation
        operationKind = kind.rawValue
        return operation.id
    }

    private func isCurrent(_ token: UUID) -> Bool { op?.id == token }

    private func release(_ token: UUID) {
        guard op?.id == token else { return }
        op = nil
        operationKind = nil
        opTask = nil
    }

    // MARK: Public operations

    /// Adoption at launch is brief and competes with nothing; a request that arrives meanwhile waits for it.
    private func settleAdoption() async {
        var waited = 0
        while op?.kind == .adopt, waited < 600 { try? await Task.sleep(nanoseconds: 20_000_000); waited += 1 }
    }

    @discardableResult
    public func start(configId: String, allowDownload: Bool) async throws -> ManagerStatus {
        await settleAdoption()
        guard let config = store.config(id: configId) else {
            throw AppError("unknown_config", "No configuration has the id \(configId).", status: 404)
        }
        let spec = EffectiveSpec(config: config, settings: store.settings)
        if let current = op {
            if current.kind == .start, current.spec == spec { return snapshot() }
            throw AppError("busy", "Another operation is in progress: \(current.kind.rawValue).", details: ["operation": current.kind.rawValue])
        }
        if let existing = try admission(for: config, spec: spec) { return existing }
        let token = try claim(.start, spec: spec)
        defer { release(token) }
        // Claim the single process slot before any suspension.
        retryTask?.cancel(); retryTask = nil; retry = nil
        state = .starting
        ownership = .managed
        detail = "Preparing"
        lastError = nil
        do {
            try await launch(config, spec: spec, allowDownload: allowDownload, token: token)
        } catch {
            if isCurrent(token), identity == nil { clearRun() ; state = .stopped; ownership = .none }
            if let app = error as? AppError { lastError = ErrorRecord(code: app.code, message: app.message, at: Date()) }
            throw error
        }
        return snapshot()
    }

    /// Stops the managed process. Without `force` it drains: new calls are refused, running
    /// calls finish, then Splash exits. It returns at once; watch `state`. `force` stops now
    /// and is for a person's explicit choice; it ends running calls.
    @discardableResult
    public func stop(force: Bool = false) async throws -> ManagerStatus {
        await settleAdoption()
        if ownership == .external {
            throw AppError("not_managed", "A Splash that this app did not start runs on this port. This app never stops it.")
        }
        if ownership == .none {
            if op == nil || force { retryTask?.cancel(); retryTask = nil; retry = nil }
            if state == .failed, op == nil { state = .stopped; detail = nil }
            if let current = op, !force, current.kind != .retry {
                throw AppError("busy", "Another operation is in progress: \(current.kind.rawValue).", details: ["operation": current.kind.rawValue])
            }
            if force, op != nil { clearRun(); op = nil; operationKind = nil; state = .stopped; detail = nil }
            return snapshot()
        }
        if let current = op {
            if current.kind == .stop, !force { return snapshot() }   // a second stop joins the first
            guard force else {
                throw AppError("busy", "Another operation is in progress: \(current.kind.rawValue).", details: ["operation": current.kind.rawValue])
            }
        }
        let id = identity
        // Claim first; everything after this may suspend.
        let token = force ? takeOver(.stop, spec: nil) : try claim(.stop, spec: nil)
        var graceful = false
        if !force {
            await ensureDrainKnown(id)
            guard isCurrent(token), identity == id else { throw AppError("busy", "The process changed while the stop was prepared.") }
            graceful = canDrain(id)
            if !graceful {
                release(token)
                throw AppError("drain_unsupported",
                               "The running Splash cannot drain, so a stop could cut a call that arrives at the same moment. Stop it from the app window with an explicit confirmation.",
                               details: ["drain_supported": drainSupported.map(String.init) ?? "unknown"])
            }
        }
        state = .stopping
        detail = graceful ? "Draining: no new calls, running calls finish" : "Stopping"
        let work: @MainActor () async -> Void = { [self] in
            defer { release(token) }
            if let hook = beforeStopSignal { await hook() }
            guard isCurrent(token) else { return }
            if let id {
                guard await stopProcess(id, graceful: graceful, token: token) else { return }
            }
            guard isCurrent(token) else { return }
            markStopped()
            try? FileManager.default.removeItem(at: paths.consoleFile)
        }
        if force { await work() } else { opTask = Task { await work() } }
        return snapshot()
    }

    /// Loads another configuration, or restarts the running one with changed saved settings.
    /// Like stop, it drains first and returns at once; watch `state`.
    @discardableResult
    public func switchTo(configId: String, allowDownload: Bool, force: Bool = false) async throws -> ManagerStatus {
        await settleAdoption()
        guard let target = store.config(id: configId) else {
            throw AppError("unknown_config", "No configuration has the id \(configId).", status: 404)
        }
        let spec = EffectiveSpec(config: target, settings: store.settings)
        if let current = op {
            if current.kind == .switchTo, current.spec == spec, !force { return snapshot() }
            if !(force && current.kind != .adopt) {
                throw AppError("busy", "Another operation is in progress: \(current.kind.rawValue).", details: ["operation": current.kind.rawValue])
            }
        }
        if ownership == .external {
            throw AppError("not_managed", "A Splash that this app did not start runs on this port. This app never stops it.")
        }
        if ownership == .none { return try await start(configId: configId, allowDownload: allowDownload) }
        if let current = applied, current.configId == target.id, current.spec == spec { return snapshot() }
        let assessment = assess(target)
        if assessment.availability == .notLocal, !allowDownload { throw downloadRequired(target, assessment) }
        let id = identity
        let token = force ? takeOver(.switchTo, spec: spec) : try claim(.switchTo, spec: spec)
        var graceful = false
        if !force {
            await ensureDrainKnown(id)
            guard isCurrent(token), identity == id else { throw AppError("busy", "The process changed while the switch was prepared.") }
            graceful = canDrain(id)
            if !graceful {
                release(token)
                throw AppError("drain_unsupported",
                               "The running Splash cannot drain, so the model is not switched automatically. Switch it from the app window with an explicit confirmation.",
                               details: ["drain_supported": drainSupported.map(String.init) ?? "unknown"])
            }
        }
        retryTask?.cancel(); retryTask = nil; retry = nil
        state = .stopping
        detail = graceful ? "Switching to \(target.displayName): draining" : "Switching to \(target.displayName)"
        let work: @MainActor () async -> Void = { [self] in
            defer { release(token) }
            if let hook = beforeStopSignal { await hook() }
            guard isCurrent(token) else { return }
            if let id {
                guard await stopProcess(id, graceful: graceful, token: token) else { return }
            }
            guard isCurrent(token) else { return }
            markStopped()
            try? FileManager.default.removeItem(at: paths.consoleFile)
            state = .starting; ownership = .managed; lastError = nil
            detail = "Preparing"
            do { try await launch(target, spec: spec, allowDownload: allowDownload, token: token) }
            catch {
                guard isCurrent(token) else { return }
                let message = (error as? AppError)?.message ?? "\(error)"
                lastError = ErrorRecord(code: (error as? AppError)?.code ?? "switch_failed", message: message, at: Date())
                clearRun(); state = .failed; ownership = .none; detail = "The model was stopped but the new one did not start: \(message)"
            }
        }
        if force { await work() } else { opTask = Task { await work() } }
        return snapshot()
    }

    /// Waits until `state == target`, a start failed for good, or the timeout ends.
    public func wait(for target: RunState, timeout: TimeInterval) async -> ManagerStatus {
        let deadline = Date().addingTimeInterval(max(0, min(timeout, 3600)))
        while Date() < deadline {
            if state == target, op == nil || target == .stopping { break }
            if state == .failed && retry == nil && op == nil { break }
            if state == .stopped && ownership == .none && op == nil { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        return snapshot()
    }

    public func checkIdle() async -> SwitchCheck {
        guard let client = currentClient() else {
            return SwitchCheck(safe: false, reason: "Splash status cannot be read, so running calls cannot be ruled out.", activity: nil)
        }
        var last: SplashClient.Activity?
        for round in 0..<2 {
            let probe = await client.probe()
            guard let activity = probe.activity else {
                return SwitchCheck(safe: false, reason: "Splash status cannot be read, so running calls cannot be ruled out.", activity: nil)
            }
            last = activity
            if !activity.idle {
                return SwitchCheck(safe: false,
                                   reason: "\(activity.activeRequests) request(s) are running or waiting.", activity: activity)
            }
            if round == 0 { try? await Task.sleep(nanoseconds: UInt64(timing.idleRecheck * 1_000_000_000)) }
        }
        return SwitchCheck(safe: true, reason: nil, activity: last)
    }

    public func resetFailure() {
        guard state == .failed, ownership == .none, op == nil else { return }
        state = .stopped; detail = nil; retry = nil; lastError = nil
    }

    public func refreshInstall() async {
        let settings = store.settings
        guard let found = SplashLocator.resolve(preferred: settings.splashPath, preferInstalled: settings.preferInstalledSplash,
                                                bundledRoot: bundledRootOverride ?? RuntimeBundle.root()) else {
            install = nil; installInspectedFor = nil; return
        }
        if installInspectedFor == found.url, install != nil { return }
        runtimeProblem = nil
        var check: RuntimeCheck?
        if found.source == .bundled, let root = found.url.deletingLastPathComponent().deletingLastPathComponent() as URL? {
            // Hash off the main actor: a few tens of megabytes.
            check = await Task.detached { RuntimeBundle.verify(root: root) }.value
            if check?.state != .verified {
                runtimeProblem = "The runtime bundled with this app failed its integrity check (\(check?.problems.joined(separator: "; ") ?? "unknown")). Reinstall the app."
                install = nil; installInspectedFor = nil
                return
            }
        }
        var inspected = await SplashLocator.inspect(found.url, source: found.source)
        inspected?.runtime = check
        install = inspected
        installInspectedFor = install == nil ? nil : found.url
    }

    /// Tests point this at a staged runtime; the app uses `RuntimeBundle.root()`.
    public var bundledRootOverride: URL?

    public func settingsChanged() {
        logs.persistTo = store.settings.persistLogsToFile ? paths.persistentLog : nil
        assessCache.removeAll()
        installInspectedFor = nil
        Task { await refreshInstall() }
    }

    // MARK: Snapshots

    public func desiredSpec(for config: ModelConfig) -> EffectiveSpec { EffectiveSpec(config: config, settings: store.settings) }

    /// Saved changes the running process does not have: edits of its configuration and of the
    /// endpoint settings. Empty when nothing runs.
    public func pendingChanges() -> [String] {
        guard ownership == .managed, let applied else { return [] }
        guard let config = store.config(id: applied.configId) else { return ["configuration \(applied.configId) was removed"] }
        return applied.spec.differences(from: desiredSpec(for: config))
    }

    public func snapshot() -> ManagerStatus {
        let settings = store.settings
        let tailnet = NetInfo.tailnetAddress()
        // The endpoint of a running process is the one it was started with.
        let port = applied?.spec.port ?? settings.inferencePort
        let exposure = applied?.spec.exposure ?? settings.inferenceExposure
        let key = applied?.keyRequired ?? (settings.inferenceExposure != .loopback)
        let bind = applied?.bindHost ?? bindHost(for: settings.inferenceExposure)
        let selected = ownership == .none ? store.config(id: settings.selectedConfigId ?? "") : nil
        let pending = pendingChanges()
        let alive = identity != nil
        let drainOK = ownership == .managed && drainSupported == true
        let safe = drainOK && state != .stopping
        var reason: String?
        if ownership == .managed, !drainOK {
            reason = drainSupported == nil ? "The running Splash has not reported whether it can drain."
                                           : "The running Splash cannot drain, so stop and switch are refused over the API."
        } else if ownership != .managed {
            reason = ownership == .external ? "Splash was not started by this app." : "Splash is not running."
        }
        let ref = (applied.map { ($0.configId, $0.displayName, $0.spec.modelId, $0.spec.revision) }
                   ?? selected.map { ($0.id, $0.displayName, $0.modelId, $0.revision) })
        return ManagerStatus(
            manager: .init(version: Self.managerVersion, startedAt: startedAtApp),
            state: state, ownership: ownership, detail: detail,
            splash: .init(installed: install != nil, version: install?.version, source: install?.source,
                          integrity: install?.runtime.map { $0.state.rawValue }, drainDeclared: install?.runtime?.drainDeclared,
                          problem: runtimeProblem),
            config: ref.map { .init(id: $0.0, displayName: $0.1, modelId: $0.2, revision: $0.3) },
            applied: applied.map {
                .init(configId: $0.configId, modelId: $0.spec.modelId, revision: $0.spec.revision,
                      options: Self.optionsView($0.spec.options), port: $0.spec.port, exposure: $0.spec.exposure,
                      allowedHosts: $0.effectiveAllowedHosts, offline: $0.offline)
            },
            pendingChanges: .init(restartRequired: !pending.isEmpty, changes: pending),
            drain: .init(supported: ownership == .managed ? drainSupported : nil, draining: draining),
            operation: operationKind,
            loadedModelId: loadedModelId,
            endpoint: .init(bindHost: bind, port: port, exposure: exposure,
                            requiresApiKey: key, localUrl: "http://127.0.0.1:\(port)",
                            tailnetUrl: exposure == .allInterfaces ? tailnet.map { "http://\($0):\(port)" } : nil,
                            openaiBasePath: "/v1",
                            allowedHosts: applied?.effectiveAllowedHosts ?? []),
            readiness: .init(processAlive: alive || ownership == .external, httpReady: httpReady, modelLoaded: modelLoaded),
            activity: .init(activeRequests: activity?.activeRequests, idle: activity?.idle, switchSafe: safe, reason: reason),
            process: identity.flatMap { id in startedAt.map {
                .init(pid: id.pid, startedAt: $0, uptimeSeconds: max(0, Int(Date().timeIntervalSince($0))), adopted: adopted) } },
            lastError: lastError, retry: retry, conflict: conflict)
    }

    static func optionsView(_ o: ServeOptions) -> ConfigView.Options {
        .init(languageOnly: o.languageOnly, maxContext: o.maxContext, maxMemory: o.maxMemory, idleRelease: o.idleRelease,
              reasoningDefault: o.reasoning.rawValue, disableAne: o.disableANE)
    }

    /// The config being run, for UIs that need a ModelConfig.
    public var activeConfig: ModelConfig? { applied?.modelConfig }

    public func configViews() -> [ConfigView] {
        let selected = store.settings.selectedConfigId
        return store.configs.map { config in
            let spec = desiredSpec(for: config)
            let isActive = ownership != .none && applied?.configId == config.id
            let assessment = assess(config)
            let runs = isActive && applied?.spec == spec
            let loaded = runs && state == .ready && modelLoaded
            return ConfigView(
                id: config.id, displayName: config.displayName, modelId: config.modelId, revision: config.revision,
                availability: loaded ? .loaded : assessment.availability,
                selected: selected == config.id, active: isActive,
                missing: assessment.missing, downloadRequired: assessment.availability == .notLocal,
                incomplete: assessment.incomplete, draftSource: assessment.draftSource.rawValue,
                verifiedStartAt: store.verifiedStart(modelId: config.modelId, revision: config.revision),
                note: isActive && !runs ? "Saved changes are not applied yet; switch to restart with them." : assessment.note,
                options: Self.optionsView(config.options))
        }
    }

    public func assess(_ config: ModelConfig) -> HFCache.Assessment {
        let families = familiesOverride ?? install?.families
        let key = "\(config.modelId)@\(config.revision ?? "")"
        if let cached = assessCache[key], Date().timeIntervalSince(cached.0) < 5 { return cached.1 }
        let value = cache.assess(modelId: config.modelId, revision: config.revision, families: families)
        assessCache[key] = (Date(), value)
        return value
    }

    // MARK: Start

    private func admission(for config: ModelConfig, spec: EffectiveSpec) throws -> ManagerStatus? {
        switch (ownership, state) {
        case (.managed, .ready), (.managed, .starting):
            guard let current = applied else { return nil }
            if current.configId == config.id {
                if current.spec == spec { return snapshot() }
                throw AppError("configuration_changed",
                               "The saved configuration differs from the running one. Use switch to restart with it.",
                               details: ["changes": current.spec.differences(from: spec).joined(separator: "; ")])
            }
            throw AppError("different_model_active",
                           "\(current.displayName) is \(state.rawValue). Use switch to change the model.",
                           details: ["active_config_id": current.configId])
        case (_, .stopping):
            throw AppError("busy", "Splash is stopping. Wait, then try again.")
        case (.external, _):
            if conflict?.modelId == config.modelId, state == .ready { return snapshot() }
            throw AppError("external_instance",
                           "A Splash that this app did not start already serves on port \(store.settings.inferencePort). Stop it yourself, or change the port.",
                           details: ["model_id": conflict?.modelId ?? ""])
        default: return nil
        }
    }

    private func launch(_ config: ModelConfig, spec: EffectiveSpec, allowDownload: Bool, token: UUID) async throws {
        await refreshInstall()
        guard isCurrent(token) else { throw AppError("cancelled", "The operation was replaced.") }
        guard let install else {
            throw AppError("splash_not_installed", "No Splash executable was found. Install it with Homebrew or set its path in Settings.", status: 424)
        }
        let assessment = assess(config)
        if assessment.availability == .notLocal, !allowDownload { throw downloadRequired(config, assessment) }

        var host = "127.0.0.1"
        var key: String?
        switch spec.exposure {
        case .loopback: break
        case .tailnet:
            // A Mac cannot connect to its own Tailscale address here, so readiness could not be proven.
            throw AppError("exposure_unsupported", "Inference cannot be bound to the Tailscale address alone. Use all interfaces with an API key.", status: 422)
        case .allInterfaces:
            host = "0.0.0.0"
            key = try Secrets.ensure(SecretAccount.inferenceKey, in: secrets)
        }
        // Splash refuses a Host name it was not told about. Over the network, accept this Mac's Tailscale names.
        var hosts = spec.allowedHosts
        if spec.exposure == .allInterfaces {
            if let address = NetInfo.tailnetAddress() { hosts.append(address) }
            if let name = await NetInfo.tailnetHostName() { hosts.append(name) }
        }
        var seen = Set<String>()
        hosts = hosts.filter { seen.insert($0).inserted }
        guard isCurrent(token) else { throw AppError("cancelled", "The operation was replaced.") }
        var runSettings = store.settings
        runSettings.inferencePort = spec.port
        runSettings.allowedHosts = hosts
        var runConfig = config
        runConfig.options = spec.options
        runConfig.revision = spec.revision
        runConfig.modelId = spec.modelId
        let arguments = try ServeArguments.build(config: runConfig, settings: runSettings, install: install, host: host,
                                                 offline: assessment.availability == .local)

        // The port must be free. Look before spawning; never touch what is there.
        if await NetInfo.isListening(host: "127.0.0.1", port: spec.port) {
            let probe = await SplashClient(host: "127.0.0.1", port: spec.port, apiKey: key ?? secrets.read(SecretAccount.inferenceKey)).probe()
            if probe.statusReadable || probe.httpReady {
                throw AppError("external_instance",
                               "A Splash that this app did not start already serves on port \(spec.port) (\(probe.instanceModel ?? "model unknown")). It was not touched.",
                               details: ["model_id": probe.instanceModel ?? "", "pid": probe.instancePid.map(String.init) ?? ""])
            }
            throw AppError("port_in_use", "Another program already listens on port \(spec.port). It was not touched.")
        }
        guard isCurrent(token) else { throw AppError("cancelled", "The operation was replaced.") }

        var environment = Self.childEnvironment()
        if let key { environment["SPLASH_API_KEY"] = key }
        try paths.prepare()
        tail.reset()
        log("Starting \(spec.modelId)\(spec.revision.map { "@\($0.prefix(8))" } ?? "") on \(host):\(spec.port)\(arguments.contains("--offline") ? " (offline, local files)" : " (may download)")")
        let spawned = try Spawner.spawn(executable: install.executable, arguments: arguments, environment: environment,
                                        output: paths.consoleFile)
        let run = AppliedConfig(configId: config.id, displayName: config.displayName, spec: spec, bindHost: host,
                                effectiveAllowedHosts: hosts, offline: arguments.contains("--offline"), keyRequired: key != nil)
        identity = spawned
        pid = spawned.pid
        adopted = false
        applied = run
        runKey = key
        readyOnce = false
        readyAt = nil
        failedProbes = 0
        draining = false
        drainSupported = nil
        startedAt = Date()
        startDeadline = Date().addingTimeInterval(allowDownload && assessment.availability == .notLocal
            ? timing.startTimeoutDownload : timing.startTimeoutLocal)
        desiredRunning = true
        detail = assessment.availability == .notLocal ? "Starting; files may be downloaded" : "Loading model"
        persistManaged(run, identity: spawned)
    }

    private func downloadRequired(_ config: ModelConfig, _ assessment: HFCache.Assessment) -> AppError {
        AppError("download_required",
                 "Starting \(config.displayName) needs files that are not complete in the local cache: \(assessment.missing.joined(separator: ", ")). Confirm the download first.",
                 details: ["missing": assessment.missing.joined(separator: ","), "config_id": config.id])
    }

    static func childEnvironment() -> [String: String] {
        let source = ProcessInfo.processInfo.environment
        var environment = ["PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"]
        for name in ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT", "HF_TOKEN"] {
            if let value = source[name] { environment[name] = value }
        }
        return environment
    }

    // MARK: Stop

    /// Reads the running Splash's status once if it has not yet said whether it can drain.
    private func ensureDrainKnown(_ id: ProcessIdentity?) async {
        guard id != nil, drainSupported == nil, readyOnce, let client = currentClient() else { return }
        let probe = await client.probe()
        if probe.statusReadable, identity == id { drainSupported = probe.draining != nil }
    }

    /// Draining is possible when the running Splash said so in its status. A process that has never
    /// been ready accepted no call, so it may be stopped at once.
    private func canDrain(_ id: ProcessIdentity?) -> Bool {
        guard id != nil else { return true }
        if drainSupported == true { return true }
        return !readyOnce && !adopted
    }

    /// Ends the process and reports whether it ended while `token` still held the slot.
    /// Graceful: ask Splash to drain and wait, without a time limit while a call is running.
    private func stopProcess(_ id: ProcessIdentity, graceful: Bool, token: UUID) async -> Bool {
        desiredRunning = false
        log("Stopping Splash (pid \(id.pid)): \(graceful && drainSupported == true ? "drain" : "now")")
        if graceful, drainSupported == true {
            Spawner.signal(id, SIGUSR1)
            draining = true
            var idleSince: Date?
            var lastProbe = Date.distantPast
            while true {
                try? await Task.sleep(nanoseconds: UInt64(timing.drainPoll * 1_000_000_000))
                guard isCurrent(token), identity == id else { return false }
                drainConsole()
                if !isRunning(id) { return true }
                if Date().timeIntervalSince(lastProbe) >= 1 {
                    lastProbe = Date()
                    let probe = await SplashClient(host: "127.0.0.1", port: applied?.spec.port ?? store.settings.inferencePort, apiKey: runKey).probe()
                    guard isCurrent(token), identity == id else { return false }
                    if let a = probe.activity {
                        activity = a
                        detail = a.idle ? "Draining: stopping" : "Draining: waiting for \(a.activeRequests) running request(s)"
                    }
                    // A call that still runs is never a reason to force. Only an idle process that
                    // does not exit is.
                    if probe.activity?.idle ?? !probe.statusReadable { idleSince = idleSince ?? Date() } else { idleSince = nil }
                    if let idleSince, Date().timeIntervalSince(idleSince) > timing.drainIdleEscalation { break }
                }
            }
            log("Splash is idle but did not exit; stopping it by signal")
        }
        return await terminate(id, token: token)
    }

    /// SIGTERM, then SIGINT, then SIGKILL. Ends running calls.
    private func terminate(_ id: ProcessIdentity, token: UUID) async -> Bool {
        Spawner.signal(id, SIGTERM)
        for (limit, next) in [(timing.termGrace, SIGINT), (timing.intGrace, SIGKILL)] {
            if await waitForExit(id, seconds: limit, token: token) { drainConsole(); return isCurrent(token) }
            guard isCurrent(token) else { return false }
            log(next == SIGINT ? "Splash did not exit after SIGTERM; sending SIGINT to stop the engine at once" : "Splash still runs; sending SIGKILL")
            Spawner.signal(id, next)
        }
        _ = await waitForExit(id, seconds: 5, token: token)
        drainConsole()
        return isCurrent(token)
    }

    private func waitForExit(_ id: ProcessIdentity, seconds: TimeInterval, token: UUID) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if !isRunning(id) { return true }
            drainConsole()
            try? await Task.sleep(nanoseconds: 100_000_000)
            if !isCurrent(token) { return false }
        }
        return !isRunning(id)
    }

    /// Reaps our own child, or checks the identity of an adopted one.
    private func isRunning(_ id: ProcessIdentity) -> Bool {
        if !adopted, Spawner.reap(id.pid) != nil { return false }
        return Spawner.isAlive(id) && !isZombie(id.pid)
    }

    private func isZombie(_ pid: Int32) -> Bool {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.size
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        guard sysctl(&mib, 4, &info, &size, nil, 0) == 0, size > 0 else { return false }
        return info.kp_proc.p_stat == SZOMB
    }

    private func clearRun() {
        identity = nil; pid = nil; startedAt = nil; adopted = false
        httpReady = false; modelLoaded = false; loadedModelId = nil; activity = nil
        applied = nil; runKey = nil; draining = false; drainSupported = nil
        try? FileManager.default.removeItem(at: paths.managedFile)
    }

    private func markStopped() {
        clearRun()
        state = .stopped; ownership = .none; detail = nil
    }

    // MARK: Polling

    func tick() async {
        guard !ticking else { return }
        ticking = true
        defer { ticking = false }
        tailnetAddress = NetInfo.tailnetAddress()
        drainConsole()
        if identity != nil { await tickManaged() } else if op == nil { await tickUnmanaged() }
    }

    private func tickManaged() async {
        // A stop or switch watches its own process; so does nothing else while it runs.
        if let kind = op?.kind, kind == .stop || kind == .switchTo { return }
        guard let id = identity, let run = applied, state != .stopping else { return }
        if !isRunning(id) { handleExit(id); return }
        let probe = await SplashClient(host: "127.0.0.1", port: run.spec.port, apiKey: runKey).probe()
        // The answer belongs to the process asked, and only while it is still the one we run.
        guard identity == id, applied == run, state != .stopping else { return }
        activity = probe.activity
        httpReady = probe.httpReady
        loadedModelId = probe.loadedModelIds.first
        modelLoaded = probe.loadedModelIds.contains(run.spec.modelId)
        if probe.statusReadable { drainSupported = probe.draining != nil; if let flag = probe.draining { draining = flag } }
        let isReady = probe.httpReady && modelLoaded
        if isReady {
            failedProbes = 0
            if state != .ready {
                state = .ready; detail = nil
                if !readyOnce {
                    readyOnce = true
                    log("Ready: \(run.spec.modelId)")
                    if !adopted { store.markVerified(modelId: run.spec.modelId, revision: run.spec.revision) }
                }
                readyAt = Date()
            }
            if let readyAt, failureCount > 0, Date().timeIntervalSince(readyAt) > timing.stableAfter { failureCount = 0 }
        } else if state == .ready {
            failedProbes += 1
            if failedProbes >= timing.notReadyTicksBeforeStarting {
                state = .starting
                detail = "Splash stopped answering; its engine may be restarting"
            }
        } else if Date() > startDeadline, op == nil {
            lastError = ErrorRecord(code: "start_timeout", message: "Splash was not ready in time. Last output: \(lastOutputSummary())", at: Date())
            log("Start timed out")
            desiredRunning = false
            let token = takeOver(.stop, spec: nil)
            state = .stopping
            _ = await terminate(id, token: token)
            if isCurrent(token) {
                let message = lastError?.message
                clearRun(); ownership = .none; state = .failed; detail = message
                release(token)
            }
        }
    }

    private func tickUnmanaged() async {
        let settings = store.settings
        let probeHost = "127.0.0.1"
        let listening = await NetInfo.isListening(host: probeHost, port: settings.inferencePort, timeout: 0.5)
        guard identity == nil, op == nil else { return }
        if !listening {
            conflict = nil
            if ownership == .external {
                ownership = .none; state = .stopped; detail = nil
                httpReady = false; modelLoaded = false; loadedModelId = nil; activity = nil; applied = nil
            }
            return
        }
        let key = settings.inferenceExposure == .loopback ? nil : secrets.read(SecretAccount.inferenceKey)
        let probe = await SplashClient(host: probeHost, port: settings.inferencePort, apiKey: key).probe()
        guard identity == nil, op == nil else { return }
        if probe.statusReadable || probe.httpReady {
            ownership = .external
            httpReady = probe.httpReady
            loadedModelId = probe.loadedModelIds.first ?? probe.instanceModel
            modelLoaded = loadedModelId != nil
            activity = probe.activity
            state = probe.httpReady ? .ready : .starting
            conflict = .init(kind: "external_splash", pid: probe.instancePid, modelId: loadedModelId,
                             message: "A Splash that this app did not start runs on port \(settings.inferencePort). This app only watches it.")
            detail = conflict?.message
        } else {
            ownership = .none
            state = .stopped
            conflict = .init(kind: probe.authRequired ? "unreadable_service" : "foreign_service", pid: nil, modelId: nil,
                             message: probe.authRequired
                             ? "A server answers on port \(settings.inferencePort) and wants a key this app does not hold."
                             : "Another program listens on port \(settings.inferencePort).")
        }
    }

    private func handleExit(_ id: ProcessIdentity) {
        drainConsole()
        let wasReady = readyOnce
        let summary = lastOutputSummary()
        let run = applied
        clearRun()
        lastError = ErrorRecord(code: wasReady ? "crashed" : "start_failed",
                                message: wasReady ? "Splash exited unexpectedly. \(summary)" : "Splash exited before it was ready. \(summary)",
                                at: Date())
        log(lastError!.message)
        ownership = .none
        // A start that never became ready fails the same way again, so it is not retried.
        if wasReady, desiredRunning, store.settings.autoRestart, failureCount < timing.retryDelays.count, let run {
            let delay = timing.retryDelays[failureCount]
            failureCount += 1
            retry = .init(attempt: failureCount, maxAttempts: timing.retryDelays.count, nextAt: Date().addingTimeInterval(delay))
            state = .failed
            applied = run          // shown while the restart is pending
            detail = "Restarting in \(Int(delay)) s (attempt \(failureCount) of \(timing.retryDelays.count))"
            retryTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                guard !Task.isCancelled, let self else { return }
                await self.retryStart(run)
            }
        } else {
            desiredRunning = false
            state = .failed
            retry = nil
            detail = wasReady ? "Splash crashed and was not restarted again." : lastError?.message
        }
    }

    /// Restarts with the configuration that was running, not with whatever is saved now.
    private func retryStart(_ run: AppliedConfig) async {
        guard state == .failed, retry != nil else { return }
        guard let token = try? claim(.retry, spec: run.spec) else {
            // Another operation is running; look again shortly.
            retryTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled, let self else { return }
                await self.retryStart(run)
            }
            return
        }
        defer { release(token) }
        retry = nil
        state = .starting; ownership = .managed; applied = run; detail = "Restarting"
        let config = ModelConfig(id: run.configId, displayName: run.displayName, modelId: run.spec.modelId,
                                 revision: run.spec.revision, options: run.spec.options)
        do { try await launch(config, spec: run.spec, allowDownload: false, token: token) }
        catch {
            guard isCurrent(token) else { return }
            let message = (error as? AppError)?.message ?? "\(error)"
            lastError = ErrorRecord(code: "restart_failed", message: message, at: Date())
            clearRun(); state = .failed; ownership = .none; detail = message; desiredRunning = false
        }
    }

    // MARK: Adoption after an app restart

    private struct ManagedRecord: Codable {
        var identity: ProcessIdentity
        var applied: AppliedConfig
    }

    private func persistManaged(_ run: AppliedConfig, identity: ProcessIdentity) {
        if let data = try? JSONEncoder().encode(ManagedRecord(identity: identity, applied: run)) {
            try? Files.writeAtomically(data, to: paths.managedFile)
        }
    }

    /// Takes over a Splash this app started before. The applied configuration comes from the record
    /// written at start, never from the saved settings of today.
    func adoptIfPossible() async {
        guard identity == nil,
              let data = try? Data(contentsOf: paths.managedFile),
              let record = try? JSONDecoder().decode(ManagedRecord.self, from: data) else { return }
        guard Spawner.isAlive(record.identity), !isZombie(record.identity.pid) else {
            try? FileManager.default.removeItem(at: paths.managedFile); return
        }
        let key = record.applied.keyRequired ? secrets.read(SecretAccount.inferenceKey) : nil
        let probe = await SplashClient(host: "127.0.0.1", port: record.applied.spec.port, apiKey: key).probe()
        // The pid must still be the process whose own status names that pid.
        guard probe.instancePid == Int(record.identity.pid) else {
            try? FileManager.default.removeItem(at: paths.managedFile)
            log("Stale managed-process record ignored")
            return
        }
        identity = record.identity; pid = record.identity.pid; adopted = true
        applied = record.applied; runKey = key
        ownership = .managed; state = .starting; desiredRunning = true
        startedAt = Date(timeIntervalSince1970: Double(record.identity.startMicros) / 1_000_000)
        readyOnce = true; startDeadline = Date().addingTimeInterval(timing.startTimeoutLocal)
        log("Adopted running Splash (pid \(record.identity.pid)) after an app restart; its console output is not followed")
    }

    // MARK: Helpers

    private func currentClient() -> SplashClient? {
        switch ownership {
        case .managed:
            guard identity != nil, let run = applied else { return nil }
            return SplashClient(host: "127.0.0.1", port: run.spec.port, apiKey: runKey)
        case .external:
            let settings = store.settings
            let key = settings.inferenceExposure == .loopback ? nil : secrets.read(SecretAccount.inferenceKey)
            return SplashClient(host: "127.0.0.1", port: settings.inferencePort, apiKey: key)
        case .none: return nil
        }
    }

    private func bindHost(for exposure: Exposure) -> String {
        exposure == .loopback ? "127.0.0.1" : "0.0.0.0"
    }

    private func drainConsole() {
        for line in tail.readNewLines() where !line.isEmpty { logs.append(line, source: .splash) }
    }

    private func lastOutputSummary() -> String {
        let candidates = logs.lines.suffix(40).filter { $0.source == .splash }.map(\.text)
        if let hit = candidates.last(where: { $0.lowercased().contains("error") }) { return hit }
        return candidates.last ?? "No output."
    }

    private func log(_ text: String) { logs.append(text, source: .app) }
}

public enum ServeArguments {
    /// Builds the argument list from a validated config. Nothing here comes from an API caller.
    public static func build(config: ModelConfig, settings: ManagerSettings, install: SplashInstall,
                             host: String, offline: Bool) throws -> [String] {
        let config = try config.validated()
        var arguments = ["serve", "--model", config.modelId]

        func need(_ flag: String) throws {
            guard install.supports(flag) else {
                throw AppError("option_unsupported",
                               "Splash \(install.version) does not offer \(flag). Remove that option from the configuration.", status: 422)
            }
        }
        if let revision = config.revision { try need("--revision"); arguments += ["--revision", revision] }
        try need("--port"); arguments += ["--port", String(settings.inferencePort)]
        try need("--host"); arguments += ["--host", host]
        for name in settings.allowedHosts { try need("--allowed-host"); arguments += ["--allowed-host", try Validation.hostName(name)] }
        if offline { try need("--offline"); arguments.append("--offline") }
        if config.options.languageOnly { try need("--language-only"); arguments.append("--language-only") }
        if let value = config.options.maxContext { try need("--max-context"); arguments += ["--max-context", value] }
        if let value = config.options.maxMemory { try need("--max-memory"); arguments += ["--max-memory", value] }
        if let value = config.options.idleRelease { try need("--idle-release"); arguments += ["--idle-release", value] }
        if config.options.reasoning == .off { try need("--default-reasoning-effort"); arguments += ["--default-reasoning-effort", "none"] }
        if config.options.disableANE { try need("--disable-ane"); arguments.append("--disable-ane") }
        return arguments
    }
}
