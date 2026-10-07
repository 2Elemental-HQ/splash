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
    public init() {}
}

/// Owns the one `splash serve` process of this app.
///
/// Everything runs on the main actor. A state change is made before the first
/// suspension point of an operation, so two concurrent calls cannot both start
/// a process.
@MainActor
public final class SplashSupervisor: ObservableObject {
    // MARK: Published state
    @Published public private(set) var state: RunState = .stopped
    @Published public private(set) var ownership: Ownership = .none
    @Published public private(set) var detail: String?
    @Published public private(set) var activeConfig: ModelConfig?
    @Published public private(set) var loadedModelId: String?
    @Published public private(set) var httpReady = false
    @Published public private(set) var modelLoaded = false
    @Published public private(set) var activity: SplashClient.Activity?
    @Published public private(set) var lastError: ErrorRecord?
    @Published public private(set) var retry: ManagerStatus.Retry?
    @Published public private(set) var conflict: ManagerStatus.Conflict?
    @Published public private(set) var install: SplashInstall?
    @Published public private(set) var startedAt: Date?
    @Published public private(set) var pid: Int32?
    @Published public private(set) var adopted = false
    @Published public private(set) var tailnetAddress: String?

    public let store: ConfigStore
    public let logs: LogBuffer
    public let startedAtApp = Date()
    public var timing: SupervisorTiming

    private let secrets: SecretStore
    private let paths: AppPaths
    private let tail: FileTail
    private var identity: ProcessIdentity?
    private var runConfig: ModelConfig?
    private var runHost = "127.0.0.1"
    private var runKey: String?
    private var readyOnce = false
    private var readyAt: Date?
    private var runStartedAt = Date()
    private var startDeadline = Date()
    private var failedProbes = 0
    private var failureCount = 0
    private var desiredRunning = false
    private var busy = false
    private var ticking = false
    private var retryTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var installInspectedFor: URL?
    private var effectiveAllowedHosts: [String] = []
    private var assessCache: [String: (Date, HFCache.Assessment)] = [:]
    public var cache = HFCache()
    public static let managerVersion = "0.1.0"

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
        pollTask = Task { [weak self] in
            guard let self else { return }
            await self.refreshInstall()
            await self.adoptIfPossible()
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

    public func shutdownSupervisor() { pollTask?.cancel(); pollTask = nil; retryTask?.cancel() }

    // MARK: Public operations

    @discardableResult
    public func start(configId: String, allowDownload: Bool) async throws -> ManagerStatus {
        guard let config = store.config(id: configId) else {
            throw AppError("unknown_config", "No configuration has the id \(configId).", status: 404)
        }
        if let current = try admissionForStart(config) { return current }
        busy = true
        defer { busy = false }
        // Claim the single process slot before any suspension.
        retryTask?.cancel(); retryTask = nil; retry = nil
        state = .starting
        ownership = .managed
        activeConfig = config
        detail = "Preparing"
        lastError = nil
        do {
            try await launch(config, allowDownload: allowDownload)
        } catch {
            if identity == nil { state = .stopped; ownership = .none; detail = nil; activeConfig = nil }
            if let app = error as? AppError { lastError = ErrorRecord(code: app.code, message: app.message, at: Date()) }
            throw error
        }
        return snapshot()
    }

    /// Stops the managed process. Refuses while calls run unless `force`.
    @discardableResult
    public func stop(force: Bool = false) async throws -> ManagerStatus {
        switch (ownership, state) {
        case (.external, _):
            throw AppError("not_managed", "A Splash that this app did not start runs on this port. This app never stops it.")
        case (_, .stopping):
            throw AppError("busy", "Splash is already stopping.")
        case (.none, _):
            retryTask?.cancel(); retryTask = nil; retry = nil
            if state == .failed { state = .stopped; detail = nil }
            return snapshot()
        default: break
        }
        if !force {
            let check = await checkIdle()
            guard check.safe else {
                throw AppError("active_requests", check.reason ?? "Calls are running.", details: activityDetails(check.activity))
            }
        }
        await terminate(reason: "Stopped from this app")
        return snapshot()
    }

    /// Changes the loaded model. Refuses while calls run, or when no reading of the activity is possible.
    @discardableResult
    public func switchTo(configId: String, allowDownload: Bool, force: Bool = false) async throws -> ManagerStatus {
        guard let target = store.config(id: configId) else {
            throw AppError("unknown_config", "No configuration has the id \(configId).", status: 404)
        }
        if ownership == .external {
            throw AppError("not_managed", "A Splash that this app did not start runs on this port. This app never stops it.")
        }
        if ownership == .managed, state == .ready || state == .starting, activeConfig?.id == target.id {
            return snapshot()
        }
        if state == .stopping || (state == .starting && ownership == .managed) {
            throw AppError("busy", "Splash is \(state.rawValue). Wait, then try again.")
        }
        if ownership == .none {
            return try await start(configId: configId, allowDownload: allowDownload)
        }
        let assessment = assess(target)
        if assessment.availability == .notLocal, !allowDownload {
            throw downloadRequired(target, assessment)
        }
        if !force {
            let check = await checkIdle()
            guard check.safe else {
                throw AppError("active_requests", "The model was not switched. \(check.reason ?? "Calls are running.")",
                               details: activityDetails(check.activity))
            }
        }
        await terminate(reason: "Switching to \(target.displayName)")
        return try await start(configId: configId, allowDownload: allowDownload)
    }

    /// Waits until `state == target` or a terminal failure, up to `timeout`.
    public func wait(for target: RunState, timeout: TimeInterval) async -> ManagerStatus {
        let deadline = Date().addingTimeInterval(max(0, min(timeout, 600)))
        while Date() < deadline {
            if state == target { break }
            if state == .failed && retry == nil { break }
            if state == .stopped && ownership == .none && !busy { break }
            try? await Task.sleep(nanoseconds: 250_000_000)
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
        guard state == .failed, ownership == .none else { return }
        state = .stopped; detail = nil; retry = nil; lastError = nil
    }

    public func refreshInstall() async {
        let url = SplashLocator.find(preferred: store.settings.splashPath)
        guard let url else { install = nil; installInspectedFor = nil; return }
        if installInspectedFor == url, install != nil { return }
        install = await SplashLocator.inspect(url)
        installInspectedFor = install == nil ? nil : url
    }

    public func settingsChanged() {
        logs.persistTo = store.settings.persistLogsToFile ? paths.persistentLog : nil
        assessCache.removeAll()
        installInspectedFor = nil
        Task { await refreshInstall() }
    }

    // MARK: Snapshots

    public func snapshot() -> ManagerStatus {
        let settings = store.settings
        let host = bindHost(for: settings)
        let key = settings.inferenceExposure != .loopback
        let tailnet = NetInfo.tailnetAddress()
        let config = (ownership == .none ? store.config(id: settings.selectedConfigId ?? "") : activeConfig)
        let alive = identity != nil
        let safe = (activity?.idle ?? false) && state == .ready
        var reason: String?
        if !safe {
            if state != .ready { reason = "Splash is not ready." }
            else if let a = activity { reason = "\(a.activeRequests) request(s) are running or waiting." }
            else { reason = "Splash status cannot be read." }
        }
        return ManagerStatus(
            manager: .init(version: Self.managerVersion, startedAt: startedAtApp),
            state: state, ownership: ownership, detail: detail,
            splash: .init(installed: install != nil, version: install?.version),
            config: config.map { .init(id: $0.id, displayName: $0.displayName, modelId: $0.modelId, revision: $0.revision) },
            loadedModelId: loadedModelId,
            endpoint: .init(bindHost: host, port: settings.inferencePort, exposure: settings.inferenceExposure,
                            requiresApiKey: key, localUrl: "http://127.0.0.1:\(settings.inferencePort)",
                            tailnetUrl: settings.inferenceExposure == .allInterfaces ? tailnet.map { "http://\($0):\(settings.inferencePort)" } : nil,
                            openaiBasePath: "/v1",
                            allowedHosts: effectiveAllowedHosts),
            readiness: .init(processAlive: alive || ownership == .external, httpReady: httpReady, modelLoaded: modelLoaded),
            activity: .init(activeRequests: activity?.activeRequests, idle: activity?.idle, switchSafe: safe, reason: reason),
            process: identity.flatMap { id in startedAt.map {
                .init(pid: id.pid, startedAt: $0, uptimeSeconds: max(0, Int(Date().timeIntervalSince($0))), adopted: adopted) } },
            lastError: lastError, retry: retry, conflict: conflict)
    }

    public func configViews() -> [ConfigView] {
        let selected = store.settings.selectedConfigId
        return store.configs.map { config in
            let isActive = ownership != .none && activeConfig?.id == config.id
            let assessment = assess(config)
            let loaded = isActive && state == .ready && modelLoaded
            return ConfigView(
                id: config.id, displayName: config.displayName, modelId: config.modelId, revision: config.revision,
                availability: loaded ? .loaded : assessment.availability,
                selected: selected == config.id, active: isActive,
                missing: assessment.missing, downloadRequired: assessment.availability == .notLocal,
                note: assessment.note,
                options: .init(languageOnly: config.options.languageOnly, maxContext: config.options.maxContext,
                               maxMemory: config.options.maxMemory, idleRelease: config.options.idleRelease,
                               reasoningDefault: config.options.reasoning.rawValue, disableAne: config.options.disableANE))
        }
    }

    public func assess(_ config: ModelConfig) -> HFCache.Assessment {
        let key = "\(config.modelId)@\(config.revision ?? "")"
        if let cached = assessCache[key], Date().timeIntervalSince(cached.0) < 5 { return cached.1 }
        let value = cache.assess(modelId: config.modelId, revision: config.revision)
        assessCache[key] = (Date(), value)
        return value
    }

    // MARK: Start

    private func admissionForStart(_ config: ModelConfig) throws -> ManagerStatus? {
        switch (ownership, state) {
        case (.managed, .ready), (.managed, .starting):
            if activeConfig?.id == config.id { return snapshot() }
            throw AppError("different_model_active",
                           "\(activeConfig?.displayName ?? "Another model") is \(state.rawValue). Use switch to change the model.",
                           details: ["active_config_id": activeConfig?.id ?? ""])
        case (_, .stopping):
            throw AppError("busy", "Splash is stopping. Wait, then try again.")
        case (.external, _):
            if conflict?.modelId == config.modelId, state == .ready { return snapshot() }
            throw AppError("external_instance",
                           "A Splash that this app did not start already serves on port \(store.settings.inferencePort). Stop it yourself, or change the port.",
                           details: ["model_id": conflict?.modelId ?? ""])
        default:
            if busy { throw AppError("busy", "Another start or stop is in progress.") }
            return nil
        }
    }

    private func launch(_ config: ModelConfig, allowDownload: Bool) async throws {
        await refreshInstall()
        guard let install else {
            throw AppError("splash_not_installed", "No Splash executable was found. Install it with Homebrew or set its path in Settings.", status: 424)
        }
        let settings = store.settings
        let assessment = assess(config)
        if assessment.availability == .notLocal, !allowDownload { throw downloadRequired(config, assessment) }

        var host = "127.0.0.1"
        var key: String?
        switch settings.inferenceExposure {
        case .loopback: break
        case .tailnet:
            // A Mac cannot connect to its own Tailscale address, so readiness could not be proven.
            throw AppError("exposure_unsupported", "Inference cannot be bound to the Tailscale address alone. Use all interfaces with an API key.", status: 422)
        case .allInterfaces:
            host = "0.0.0.0"
            key = try Secrets.ensure(SecretAccount.inferenceKey, in: secrets)
        }
        // Splash refuses a Host name it was not told about. Over the network, accept this Mac's Tailscale names.
        var hostSettings = settings
        if settings.inferenceExposure == .allInterfaces {
            var hosts = settings.allowedHosts
            if let address = NetInfo.tailnetAddress() { hosts.append(address) }
            if let name = await NetInfo.tailnetHostName() { hosts.append(name) }
            var seen = Set<String>()
            hostSettings.allowedHosts = hosts.filter { seen.insert($0).inserted }
        }
        effectiveAllowedHosts = hostSettings.allowedHosts
        let arguments = try ServeArguments.build(config: config, settings: hostSettings, install: install, host: host,
                                                 offline: assessment.availability == .local)

        // The port must be free. Look before spawning; never touch what is there.
        let probeHost = host == "0.0.0.0" ? "127.0.0.1" : host
        if await NetInfo.isListening(host: probeHost, port: settings.inferencePort) {
            let client = SplashClient(host: probeHost, port: settings.inferencePort, apiKey: key ?? secrets.read(SecretAccount.inferenceKey))
            let probe = await client.probe()
            if probe.statusReadable || probe.httpReady {
                throw AppError("external_instance",
                               "A Splash that this app did not start already serves on port \(settings.inferencePort) (\(probe.instanceModel ?? "model unknown")). It was not touched.",
                               details: ["model_id": probe.instanceModel ?? "", "pid": probe.instancePid.map(String.init) ?? ""])
            }
            throw AppError("port_in_use", "Another program already listens on port \(settings.inferencePort). It was not touched.")
        }

        var environment = Self.childEnvironment()
        if let key { environment["SPLASH_API_KEY"] = key }
        try paths.prepare()
        tail.reset()
        log("Starting \(config.modelId)\(config.revision.map { "@\($0.prefix(8))" } ?? "") on \(host):\(settings.inferencePort)\(arguments.contains("--offline") ? " (offline, local files)" : " (may download)")")
        let spawned = try Spawner.spawn(executable: install.executable, arguments: arguments, environment: environment,
                                        output: paths.consoleFile)
        identity = spawned
        pid = spawned.pid
        adopted = false
        runConfig = config
        runHost = probeHost
        runKey = key
        readyOnce = false
        readyAt = nil
        failedProbes = 0
        runStartedAt = Date()
        startedAt = runStartedAt
        startDeadline = runStartedAt.addingTimeInterval(allowDownload && assessment.availability == .notLocal
            ? timing.startTimeoutDownload : timing.startTimeoutLocal)
        desiredRunning = true
        detail = assessment.availability == .notLocal ? "Starting; files may be downloaded" : "Loading model"
        persistManaged(config: config, identity: spawned)
    }

    private func downloadRequired(_ config: ModelConfig, _ assessment: HFCache.Assessment) -> AppError {
        AppError("download_required",
                 "Starting \(config.displayName) needs files that are not in the local cache: \(assessment.missing.joined(separator: ", ")). Confirm the download first.",
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

    private func terminate(reason: String) async {
        guard let id = identity else { markStopped(); return }
        busy = true
        defer { busy = false }
        desiredRunning = false
        retryTask?.cancel(); retryTask = nil; retry = nil
        state = .stopping
        detail = reason
        log("Stopping Splash (pid \(id.pid)): \(reason)")
        Spawner.signal(id, SIGTERM)
        if await waitForExit(id, seconds: timing.termGrace) == false {
            log("Splash did not exit after SIGTERM; sending SIGINT to stop the engine at once")
            Spawner.signal(id, SIGINT)
            if await waitForExit(id, seconds: timing.intGrace) == false {
                log("Splash still runs; sending SIGKILL")
                Spawner.signal(id, SIGKILL)
                _ = await waitForExit(id, seconds: 5)
            }
        }
        drainConsole()
        markStopped()
        try? FileManager.default.removeItem(at: paths.consoleFile)
    }

    private func waitForExit(_ id: ProcessIdentity, seconds: TimeInterval) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if !isRunning(id) { return true }
            drainConsole()
            try? await Task.sleep(nanoseconds: 100_000_000)
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

    private func markStopped() {
        identity = nil; pid = nil; startedAt = nil; adopted = false
        state = .stopped; ownership = .none; detail = nil
        httpReady = false; modelLoaded = false; loadedModelId = nil; activity = nil
        activeConfig = nil
        try? FileManager.default.removeItem(at: paths.managedFile)
    }

    // MARK: Polling

    func tick() async {
        guard !ticking else { return }
        ticking = true
        defer { ticking = false }
        tailnetAddress = NetInfo.tailnetAddress()
        drainConsole()
        if identity != nil { await tickManaged() } else if !busy { await tickUnmanaged() }
    }

    private func tickManaged() async {
        guard let id = identity, let config = runConfig ?? activeConfig, state != .stopping else { return }
        if !isRunning(id) { handleExit(id); return }
        let client = SplashClient(host: runHost, port: store.settings.inferencePort, apiKey: runKey)
        let probe = await client.probe()
        guard identity == id else { return }   // changed while waiting
        activity = probe.activity
        httpReady = probe.httpReady
        loadedModelId = probe.loadedModelIds.first
        modelLoaded = probe.loadedModelIds.contains(config.modelId)
        let isReady = probe.httpReady && modelLoaded
        if isReady {
            failedProbes = 0
            if state != .ready {
                state = .ready; detail = nil
                if !readyOnce { readyOnce = true; log("Ready: \(config.modelId)") }
                readyAt = Date()
            }
            if let readyAt, failureCount > 0, Date().timeIntervalSince(readyAt) > timing.stableAfter { failureCount = 0 }
        } else if state == .ready {
            failedProbes += 1
            if failedProbes >= timing.notReadyTicksBeforeStarting {
                state = .starting
                detail = "Splash stopped answering; its engine may be restarting"
            }
        } else if Date() > startDeadline {
            lastError = ErrorRecord(code: "start_timeout", message: "Splash was not ready in time. Last output: \(lastOutputSummary())", at: Date())
            log("Start timed out")
            desiredRunning = false
            await terminate(reason: "Start timed out")
            state = .failed
            detail = lastError?.message
        }
    }

    private func tickUnmanaged() async {
        let settings = store.settings
        let probeHost = "127.0.0.1"
        let listening = await NetInfo.isListening(host: probeHost, port: settings.inferencePort, timeout: 0.5)
        guard identity == nil, !busy else { return }
        if !listening {
            conflict = nil
            if ownership == .external {
                ownership = .none; state = .stopped; detail = nil
                httpReady = false; modelLoaded = false; loadedModelId = nil; activity = nil; activeConfig = nil
            }
            return
        }
        let key = settings.inferenceExposure == .loopback ? nil : secrets.read(SecretAccount.inferenceKey)
        let probe = await SplashClient(host: probeHost, port: settings.inferencePort, apiKey: key).probe()
        guard identity == nil, !busy else { return }
        if probe.statusReadable || probe.httpReady {
            ownership = .external
            httpReady = probe.httpReady
            loadedModelId = probe.loadedModelIds.first ?? probe.instanceModel
            modelLoaded = loadedModelId != nil
            activity = probe.activity
            state = probe.httpReady ? .ready : .starting
            activeConfig = store.configs.first { $0.modelId == loadedModelId }
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
        let config = runConfig
        identity = nil; pid = nil; adopted = false
        httpReady = false; modelLoaded = false; loadedModelId = nil; activity = nil
        try? FileManager.default.removeItem(at: paths.managedFile)
        lastError = ErrorRecord(code: wasReady ? "crashed" : "start_failed",
                                message: wasReady ? "Splash exited unexpectedly. \(summary)" : "Splash exited before it was ready. \(summary)",
                                at: Date())
        log(lastError!.message)
        ownership = .none
        // A start that never became ready fails the same way again, so it is not retried.
        if wasReady, desiredRunning, store.settings.autoRestart, failureCount < timing.retryDelays.count, let config {
            let delay = timing.retryDelays[failureCount]
            failureCount += 1
            let next = Date().addingTimeInterval(delay)
            retry = .init(attempt: failureCount, maxAttempts: timing.retryDelays.count, nextAt: next)
            state = .failed
            activeConfig = config
            detail = "Restarting in \(Int(delay)) s (attempt \(failureCount) of \(timing.retryDelays.count))"
            retryTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                guard !Task.isCancelled, let self else { return }
                await self.retryStart(config)
            }
        } else {
            desiredRunning = false
            state = .failed
            retry = nil
            detail = wasReady ? "Splash crashed and was not restarted again." : lastError?.message
            activeConfig = config
        }
    }

    private func retryStart(_ config: ModelConfig) async {
        guard state == .failed, retry != nil, !busy else { return }
        retry = nil
        busy = true
        defer { busy = false }
        state = .starting; ownership = .managed; activeConfig = config; detail = "Restarting"
        do { try await launch(config, allowDownload: false) }
        catch {
            let message = (error as? AppError)?.message ?? "\(error)"
            lastError = ErrorRecord(code: "restart_failed", message: message, at: Date())
            state = .failed; ownership = .none; detail = message; desiredRunning = false
        }
    }

    // MARK: Adoption after an app restart

    private struct ManagedRecord: Codable {
        var identity: ProcessIdentity
        var configId: String
        var modelId: String
        var port: Int
        var exposure: Exposure
    }

    private func persistManaged(config: ModelConfig, identity: ProcessIdentity) {
        let record = ManagedRecord(identity: identity, configId: config.id, modelId: config.modelId,
                                   port: store.settings.inferencePort, exposure: store.settings.inferenceExposure)
        if let data = try? JSONEncoder().encode(record) { try? Files.writeAtomically(data, to: paths.managedFile) }
    }

    func adoptIfPossible() async {
        guard identity == nil,
              let data = try? Data(contentsOf: paths.managedFile),
              let record = try? JSONDecoder().decode(ManagedRecord.self, from: data) else { return }
        guard Spawner.isAlive(record.identity), !isZombie(record.identity.pid) else {
            try? FileManager.default.removeItem(at: paths.managedFile); return
        }
        let host: String
        var key: String?
        switch record.exposure {
        case .loopback: host = "127.0.0.1"
        case .tailnet, .allInterfaces: host = "127.0.0.1"; key = secrets.read(SecretAccount.inferenceKey)
        }
        let probe = await SplashClient(host: host, port: record.port, apiKey: key).probe()
        // The pid must still be the process whose own status names that pid.
        guard probe.instancePid == Int(record.identity.pid) else {
            try? FileManager.default.removeItem(at: paths.managedFile)
            log("Stale managed-process record ignored")
            return
        }
        let config = store.config(id: record.configId) ?? ModelConfig(id: record.configId, displayName: record.modelId, modelId: record.modelId)
        identity = record.identity; pid = record.identity.pid; adopted = true
        runConfig = config; activeConfig = config; runHost = host; runKey = key
        ownership = .managed; state = .starting; desiredRunning = true
        startedAt = Date(timeIntervalSince1970: Double(record.identity.startMicros) / 1_000_000)
        readyOnce = true; startDeadline = Date().addingTimeInterval(timing.startTimeoutLocal)
        log("Adopted running Splash (pid \(record.identity.pid)) after an app restart; its console output is not followed")
    }

    // MARK: Helpers

    private func currentClient() -> SplashClient? {
        switch ownership {
        case .managed:
            guard identity != nil else { return nil }
            return SplashClient(host: runHost, port: store.settings.inferencePort, apiKey: runKey)
        case .external:
            let settings = store.settings
            let host = "127.0.0.1"
            let key = settings.inferenceExposure == .loopback ? nil : secrets.read(SecretAccount.inferenceKey)
            return SplashClient(host: host, port: settings.inferencePort, apiKey: key)
        case .none: return nil
        }
    }

    private func bindHost(for settings: ManagerSettings) -> String {
        switch settings.inferenceExposure {
        case .loopback: return "127.0.0.1"
        case .tailnet, .allInterfaces: return "0.0.0.0"
        }
    }

    private func activityDetails(_ activity: SplashClient.Activity?) -> [String: String]? {
        activity.map { ["active_requests": String($0.activeRequests), "queued": String($0.queued), "generating": String($0.generating)] }
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
