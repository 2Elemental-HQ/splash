import XCTest
@testable import SplashManagerCore

@MainActor
final class SupervisorTests: XCTestCase {
    func start(_ h: Harness, _ id: String = "a") async throws {
        _ = try await h.supervisor.start(configId: id, allowDownload: true)
        await h.expect { h.supervisor.state == .ready }
        await h.expect { h.supervisor.drainSupported != nil }
    }

    func testStartReachesReadyOnlyWhenModelIsListed() async throws {
        let h = try Harness(control: ["ready_delay": 1.0])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        XCTAssertEqual(h.supervisor.state, .starting)
        // The process and its port exist, but /ready answers 503: not ready.
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertEqual(h.supervisor.state, .starting)
        await h.expect { h.supervisor.state == .ready }
        let status = h.supervisor.snapshot()
        XCTAssertTrue(status.readiness.httpReady && status.readiness.modelLoaded)
        XCTAssertEqual(status.loadedModelId, "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(status.ownership, .managed)
        let argv = h.startLog.first?["argv"] as? [String] ?? []
        XCTAssertTrue(argv.contains("--offline"), "a model whose files are complete starts offline")
        XCTAssertEqual(argv[argv.firstIndex(of: "--revision")! + 1], commitA)
        XCTAssertNotNil(h.store.verifiedStart(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA), "a start that reached ready is recorded")
        await h.cleanup()
    }

    func testConcurrentStartsStartOneProcess() async throws {
        let h = try Harness()
        h.supervisor.begin()
        let results = await withTaskGroup(of: String.self) { group -> [String] in
            for _ in 0..<12 {
                group.addTask { @MainActor in
                    do { _ = try await h.supervisor.start(configId: "a", allowDownload: false); return "ok" }
                    catch let e as AppError { return e.code } catch { return "other" }
                }
            }
            var out: [String] = []
            for await r in group { out.append(r) }
            return out
        }
        XCTAssertEqual(results.filter { $0 == "ok" }.count, 12, "same-config starts are idempotent: \(results)")
        await h.expect { h.supervisor.state == .ready }
        XCTAssertEqual(h.startLog.count, 1, "exactly one process was started")
        await h.cleanup()
    }

    func testStartOfOtherConfigWhileRunningIsRefused() async throws {
        let h = try Harness()
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.startLog.count == 1 }
        do { _ = try await h.supervisor.start(configId: "b", allowDownload: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "different_model_active") }
        XCTAssertEqual(h.startLog.count, 1)
        await h.cleanup()
    }

    func testDownloadNeedsConfirmation() async throws {
        let h = try Harness()
        h.supervisor.begin()
        do { _ = try await h.supervisor.start(configId: "b", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError {
            XCTAssertEqual(e.code, "download_required")
            XCTAssertTrue(e.details?["missing"]?.contains("Qwen3.6-35B-A3B-4bit") == true)
        }
        XCTAssertEqual(h.startLog.count, 0)
        XCTAssertEqual(h.supervisor.state, .stopped)
        let views = h.supervisor.configViews()
        XCTAssertEqual(views.first { $0.id == "a" }?.availability, .local)
        XCTAssertEqual(views.first { $0.id == "b" }?.availability, .notLocal)
        XCTAssertNil(views.first { $0.id == "a" }?.verifiedStartAt, "files in the cache are not a proven start")
        await h.cleanup()
    }

    func testIncompleteCacheIsNotLocal() async throws {
        let h = try Harness()
        // Remove the tokenizer from the cached snapshot: config.json and a weight file alone are not enough.
        try FileManager.default.removeItem(at: h.dir.appendingPathComponent("hf/models--mlx-community--Qwen3.8-27B-4bit/snapshots/\(commitA)/tokenizer.json"))
        let view = h.supervisor.configViews().first { $0.id == "a" }
        XCTAssertEqual(view?.availability, .notLocal)
        XCTAssertEqual(view?.incomplete["mlx-community/Qwen3.8-27B-4bit"], ["tokenizer.json"])
        do { _ = try await h.supervisor.start(configId: "a", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "download_required") }
        await h.cleanup()
    }

    func testForeignListenerOnPortIsLeftAlone() async throws {
        let h = try Harness()
        h.supervisor.begin()
        let listener = try ForeignListener(port: h.port)
        defer { listener.close() }
        do { _ = try await h.supervisor.start(configId: "a", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "port_in_use") }
        XCTAssertTrue(listener.stillOpen())
        XCTAssertEqual(h.startLog.count, 0)
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    func testExternalSplashIsRecognisedAndNeverStopped() async throws {
        let h = try Harness()
        h.supervisor.begin()
        let external = Process()
        external.executableURL = h.fake
        external.arguments = ["serve", "--model", "mlx-community/Qwen3.8-27B-4bit", "--port", String(h.port)]
        external.standardOutput = FileHandle.nullDevice
        try external.run()
        defer { if external.isRunning { external.terminate() } }
        await h.expect { h.supervisor.ownership == .external && h.supervisor.state == .ready }
        XCTAssertEqual(h.supervisor.snapshot().conflict?.kind, "external_splash")
        for force in [false, true] {
            do { _ = try await h.supervisor.stop(force: force); XCTFail("expected refusal") }
            catch let e as AppError { XCTAssertEqual(e.code, "not_managed") }
        }
        do { _ = try await h.supervisor.switchTo(configId: "a", allowDownload: true, force: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "not_managed") }
        do { _ = try await h.supervisor.start(configId: "b", allowDownload: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "external_instance") }
        let same = try await h.supervisor.start(configId: "a", allowDownload: false)
        XCTAssertEqual(same.ownership, .external)
        XCTAssertTrue(external.isRunning)
        external.terminate()
        await h.expect { h.supervisor.ownership == .none }
        await h.cleanup()
    }

    // MARK: Stop and switch

    func testStopDrainsAndNeverCutsARunningCall() async throws {
        let h = try Harness(control: ["hold": 2.0])
        h.supervisor.begin()
        try await start(h)
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        let status = try await h.supervisor.stop()
        XCTAssertEqual(status.state, .stopping)
        XCTAssertEqual(status.operation, "stop")
        // From now on a new call is refused with Splash's own code; the held one runs.
        await h.expect { h.supervisor.draining }
        let late = await h.chat()
        XCTAssertEqual(late, 503)
        XCTAssertTrue(h.supervisor.snapshot().drain.draining)
        let finished = await call.value
        XCTAssertEqual(finished, 200, "the call that was running finished normally")
        await h.expect { h.supervisor.state == .stopped }
        XCTAssertEqual(h.supervisor.ownership, .none)
        await h.cleanup()
    }

    /// The race of the review: a call arrives after the last idle reading and before the stop signal.
    func testCallAcceptedJustBeforeTheStopSignalSurvives() async throws {
        let h = try Harness(control: ["hold": 2.0])
        h.supervisor.begin()
        try await start(h)
        var call: Task<Int, Never>?
        h.supervisor.beforeStopSignal = {
            call = Task { await h.chat() }
            // Splash itself counts the call before the signal is sent.
            let client = SplashClient(host: "127.0.0.1", port: h.port, apiKey: nil)
            for _ in 0..<100 {
                if await client.probe().activity?.activeRequests == 1 { return }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        _ = try await h.supervisor.stop()
        await h.expect { h.supervisor.state == .stopped }
        let finished = await call?.value
        XCTAssertEqual(finished, 200)
        await h.cleanup()
    }

    func testForceStopEndsARunningCall() async throws {
        let h = try Harness(control: ["hold": 5.0])
        h.supervisor.begin()
        try await start(h)
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        _ = try await h.supervisor.stop(force: true)
        XCTAssertEqual(h.supervisor.state, .stopped)
        let result = await call.value
        XCTAssertNotEqual(result, 200, "force is the one path that ends a running call")
        await h.cleanup()
    }

    func testSwitchDrainsThenStartsTheOtherModel() async throws {
        let h = try Harness(control: ["hold": 2.0])
        h.supervisor.begin()
        try await start(h)
        let firstPid = h.supervisor.pid
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        let status = try await h.supervisor.switchTo(configId: "b", allowDownload: true)
        XCTAssertEqual(status.state, .stopping)
        let finished = await call.value
        XCTAssertEqual(finished, 200)
        await h.expect { h.supervisor.state == .ready && h.supervisor.loadedModelId == "mlx-community/Qwen3.6-35B-A3B-4bit" }
        XCTAssertNotEqual(firstPid, h.supervisor.pid)
        XCTAssertEqual(h.startLog.count, 2)
        XCTAssertEqual(h.supervisor.applied?.configId, "b")
        await h.cleanup()
    }

    func testWithoutDrainSupportStopAndSwitchAreRefusedAndNotCalledSafe() async throws {
        let h = try Harness(control: ["no_drain": true])
        h.supervisor.begin()
        try await start(h)
        let status = h.supervisor.snapshot()
        XCTAssertEqual(status.drain.supported, false)
        XCTAssertFalse(status.activity.switchSafe, "an idle snapshot is not a guarantee")
        XCTAssertTrue(h.supervisor.snapshot().activity.idle == true)
        do { _ = try await h.supervisor.stop(); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "drain_unsupported") }
        do { _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "drain_unsupported") }
        XCTAssertEqual(h.supervisor.state, .ready)
        XCTAssertNil(h.supervisor.snapshot().operation, "a refused request leaves no operation behind")
        // The person can still force it, explicitly.
        _ = try await h.supervisor.stop(force: true)
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    func testForceStopEscalatesWhenSigtermIsIgnored() async throws {
        var t = SupervisorTiming(); t.tick = 0.1; t.termGrace = 0.6; t.intGrace = 3; t.idleRecheck = 0.05
        let h = try Harness(control: ["ignore_term": true], timing: t)
        h.supervisor.begin()
        try await start(h)
        let identity = Spawner.identity(of: h.supervisor.pid!)!
        _ = try await h.supervisor.stop(force: true)
        XCTAssertFalse(Spawner.isAlive(identity))
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    // MARK: Lifecycle exclusivity

    func testStopAndSwitchOverlapRunOnlyOne() async throws {
        let h = try Harness(control: ["hold": 1.5])
        h.supervisor.begin()
        try await start(h)
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        let results = await withTaskGroup(of: String.self) { group -> [String] in
            group.addTask { @MainActor in
                do { _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true); return "switch ok" }
                catch let e as AppError { return "switch \(e.code)" } catch { return "switch other" }
            }
            group.addTask { @MainActor in
                do { _ = try await h.supervisor.stop(); return "stop ok" }
                catch let e as AppError { return "stop \(e.code)" } catch { return "stop other" }
            }
            var out: [String] = []
            for await r in group { out.append(r) }
            return out.sorted()
        }
        XCTAssertEqual(results.filter { $0.hasSuffix("ok") }.count, 1, "\(results)")
        XCTAssertEqual(results.filter { $0.hasSuffix("busy") }.count, 1, "\(results)")
        let finished = await call.value
        XCTAssertEqual(finished, 200)
        if results.contains("switch ok") {
            await h.expect { h.supervisor.state == .ready && h.supervisor.applied?.configId == "b" }
            XCTAssertEqual(h.startLog.count, 2)
        } else {
            await h.expect { h.supervisor.state == .stopped }
            try await Task.sleep(nanoseconds: 500_000_000)
            XCTAssertEqual(h.startLog.count, 1)
        }
        await h.cleanup()
    }

    func testTwoSwitchesToTheSameTargetJoinAndDifferentOnesDoNot() async throws {
        let h = try Harness()
        h.supervisor.begin()
        try await start(h)
        let same = await withTaskGroup(of: String.self) { group -> [String] in
            for _ in 0..<3 {
                group.addTask { @MainActor in
                    do { _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true); return "ok" }
                    catch let e as AppError { return e.code } catch { return "other" }
                }
            }
            var out: [String] = []
            for await r in group { out.append(r) }
            return out
        }
        XCTAssertEqual(same, ["ok", "ok", "ok"])
        do { _ = try await h.supervisor.switchTo(configId: "a", allowDownload: true); XCTFail("expected busy") }
        catch let e as AppError { XCTAssertEqual(e.code, "busy") }
        await h.expect { h.supervisor.state == .ready && h.supervisor.applied?.configId == "b" }
        XCTAssertEqual(h.startLog.count, 2, "one switch ran")
        await h.cleanup()
    }

    func testTwoStopsJoinAndLeaveNoStaleStopForTheNextProcess() async throws {
        let h = try Harness(control: ["hold": 1.0])
        h.supervisor.begin()
        try await start(h)
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        _ = try await h.supervisor.stop()
        _ = try await h.supervisor.stop()   // joins the first
        _ = await call.value
        await h.expect { h.supervisor.state == .stopped }
        // A new process started afterwards is not touched by the earlier requests.
        try await start(h)
        try await Task.sleep(nanoseconds: 800_000_000)
        XCTAssertEqual(h.supervisor.state, .ready)
        XCTAssertEqual(h.startLog.count, 2)
        await h.cleanup()
    }

    func testForceStopReplacesADrainingSwitchAndTheSwitchDoesNotStartAnything() async throws {
        let h = try Harness(control: ["hold": 8.0])
        h.supervisor.begin()
        try await start(h)
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true)
        XCTAssertEqual(h.supervisor.state, .stopping)
        _ = try await h.supervisor.stop(force: true)
        _ = await call.value
        try await Task.sleep(nanoseconds: 1_200_000_000)
        XCTAssertEqual(h.supervisor.state, .stopped)
        XCTAssertEqual(h.startLog.count, 1, "the replaced switch never starts its target")
        XCTAssertNil(h.supervisor.snapshot().operation)
        await h.cleanup()
    }

    // MARK: Saved against applied configuration

    func testRunningProcessKeepsItsAppliedConfigurationWhenSettingsChange() async throws {
        let h = try Harness()
        h.supervisor.begin()
        try await start(h)
        let port = h.port
        // Save a new port, a new exposure and a new revision while the process runs.
        try h.store.updateSettings { $0.inferencePort = Harness.freePort(); $0.inferenceExposure = .allInterfaces }
        var config = h.store.config(id: "a")!
        config.revision = commitC; config.options.maxContext = "64K"
        try h.store.save(config)
        try await Task.sleep(nanoseconds: 800_000_000)   // several ticks
        let status = h.supervisor.snapshot()
        XCTAssertEqual(status.state, .ready, "monitoring still follows the running process")
        XCTAssertEqual(status.endpoint.port, port)
        XCTAssertEqual(status.endpoint.exposure, .loopback)
        XCTAssertFalse(status.endpoint.requiresApiKey)
        XCTAssertEqual(status.applied?.revision, commitA)
        XCTAssertNil(status.applied?.options.maxContext)
        XCTAssertTrue(status.pendingChanges.restartRequired)
        let changes = status.pendingChanges.changes.joined(separator: "\n")
        XCTAssertTrue(changes.contains("revision"), changes)
        XCTAssertTrue(changes.contains("max context"), changes)
        XCTAssertTrue(changes.contains("inference port"), changes)
        XCTAssertTrue(changes.contains("exposure"), changes)
        await h.cleanup()
    }

    func testSameIdWithChangedSettingsIsNotIdempotentAndSwitchRestartsIt() async throws {
        let h = try Harness()
        h.supervisor.begin()
        try await start(h)
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)   // exact same: idempotent
        var config = h.store.config(id: "a")!
        config.options.maxContext = "64K"
        try h.store.save(config)
        do { _ = try await h.supervisor.start(configId: "a", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError {
            XCTAssertEqual(e.code, "configuration_changed")
            XCTAssertTrue(e.details?["changes"]?.contains("max context") == true)
        }
        let firstPid = h.supervisor.pid
        _ = try await h.supervisor.switchTo(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready && h.supervisor.pid != firstPid }
        let argv = h.startLog.last?["argv"] as? [String] ?? []
        XCTAssertEqual(argv[argv.firstIndex(of: "--max-context")! + 1], "64K")
        XCTAssertEqual(h.supervisor.applied?.spec.options.maxContext, "64K")
        XCTAssertFalse(h.supervisor.snapshot().pendingChanges.restartRequired)
        // Applied equals saved now: switching to it again is idempotent.
        _ = try await h.supervisor.switchTo(configId: "a", allowDownload: false)
        XCTAssertEqual(h.startLog.count, 2)
        await h.cleanup()
    }

    func testRevisionChangeIsAppliedByRestart() async throws {
        let h = try Harness()
        h.supervisor.begin()
        try await start(h)
        var config = h.store.config(id: "a")!
        config.revision = commitC
        try h.store.save(config)
        _ = try await h.supervisor.switchTo(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready && h.startLog.count == 2 }
        let argv = h.startLog.last?["argv"] as? [String] ?? []
        XCTAssertEqual(argv[argv.firstIndex(of: "--revision")! + 1], commitC)
        await h.cleanup()
    }

    func testAdoptionKeepsTheAppliedConfigurationNotTheSavedOne() async throws {
        let h = try Harness()
        h.supervisor.begin()
        var config = h.store.config(id: "a")!
        config.options.maxContext = "64K"
        try h.store.save(config)
        try await start(h)
        let pid = h.supervisor.pid!
        h.supervisor.shutdownSupervisor()
        // The saved settings change while the app is away.
        config.options.maxContext = "32K"; config.revision = commitC
        try h.store.save(config)
        let second = SplashSupervisor(store: h.store, secrets: h.secrets, logs: LogBuffer(), timing: h.supervisor.timing)
        second.cache = h.supervisor.cache
        second.familiesOverride = testFamilies
        second.begin()
        await h.expect { second.state == .ready && second.ownership == .managed }
        XCTAssertEqual(second.pid, pid)
        XCTAssertTrue(second.adopted)
        XCTAssertEqual(second.applied?.spec.options.maxContext, "64K")
        XCTAssertEqual(second.applied?.spec.revision, commitA)
        let status = second.snapshot()
        XCTAssertTrue(status.pendingChanges.restartRequired)
        // The same id with the saved (different) settings is not "already running".
        do { _ = try await second.start(configId: "a", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "configuration_changed") }
        _ = try await second.stop(force: true)
        _ = Spawner.reap(pid)   // here the runner is the parent; after a real app restart launchd is
        XCTAssertFalse(Spawner.isAlive(ProcessIdentity(pid: pid, startMicros: Spawner.startMicros(of: pid) ?? 0)))
        second.shutdownSupervisor()
        try? FileManager.default.removeItem(at: h.dir)
    }

    func testStalePidRecordIsNotAdopted() async throws {
        let h = try Harness()
        let me = Spawner.identity(of: getpid())!
        let applied = AppliedConfig(configId: "a", displayName: "A",
                                    spec: EffectiveSpec(config: h.store.config(id: "a")!, settings: h.store.settings),
                                    bindHost: "127.0.0.1", effectiveAllowedHosts: [], offline: true, keyRequired: false)
        let record = ["identity": ["pid": Int(me.pid), "startMicros": me.startMicros], "applied": try JSONSerialization.jsonObject(with: JSONEncoder().encode(applied))] as [String: Any]
        try h.paths.prepare()
        try JSONSerialization.data(withJSONObject: record).write(to: h.paths.managedFile)
        h.supervisor.begin()
        try await Task.sleep(nanoseconds: 800_000_000)
        XCTAssertEqual(h.supervisor.ownership, .none)
        XCTAssertFalse(FileManager.default.fileExists(atPath: h.paths.managedFile.path))
        await h.cleanup()
    }

    // MARK: Failures

    func testStartFailureIsReportedAndNotRetried() async throws {
        let h = try Harness(control: ["fail_start": true])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .failed }
        XCTAssertEqual(h.supervisor.lastError?.code, "start_failed")
        XCTAssertTrue(h.supervisor.lastError?.message.contains("cannot load the model") == true)
        try await Task.sleep(nanoseconds: 1_000_000_000)
        XCTAssertEqual(h.startLog.count, 1, "a failed start is not retried")
        XCTAssertNil(h.supervisor.retry)
        await h.cleanup()
    }

    func testCrashRestartsBoundedTimes() async throws {
        let h = try Harness(control: ["crash_after": 0.2])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect(20) { h.startLog.count == 4 && h.supervisor.state == .failed && h.supervisor.retry == nil }
        try await Task.sleep(nanoseconds: 1_500_000_000)
        XCTAssertEqual(h.startLog.count, 4, "no unbounded restart loop")
        XCTAssertEqual(h.supervisor.lastError?.code, "crashed")
        await h.cleanup()
    }

    func testRestartAfterCrashUsesTheAppliedConfigurationNotTheSavedOne() async throws {
        let h = try Harness(control: ["crash_after": 0.2, "crash_limit": 1])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.startLog.count == 1 }
        var config = h.store.config(id: "a")!
        config.options.maxContext = "16K"
        try h.store.save(config)
        await h.expect(15) { h.startLog.count == 2 && h.supervisor.state == .ready }
        let argv = h.startLog.last?["argv"] as? [String] ?? []
        XCTAssertFalse(argv.contains("--max-context"), "the restart reproduces what ran, the saved change waits")
        XCTAssertTrue(h.supervisor.snapshot().pendingChanges.restartRequired)
        await h.cleanup()
    }

    func testManualStopDuringRetryBackoffCancelsRestart() async throws {
        var t = SupervisorTiming(); t.tick = 0.1; t.retryDelays = [1.5]; t.idleRecheck = 0.05
        let h = try Harness(control: ["crash_after": 0.1], timing: t)
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.retry != nil }
        _ = try await h.supervisor.stop()
        try await Task.sleep(nanoseconds: 2_000_000_000)
        XCTAssertEqual(h.startLog.count, 1)
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    func testUnsupportedOptionIsRefusedBeforeStart() async throws {
        let h = try Harness()
        var config = h.store.config(id: "a")!
        config.options.disableANE = true   // the fake does not list --disable-ane
        try h.store.save(config)
        h.supervisor.begin()
        do { _ = try await h.supervisor.start(configId: "a", allowDownload: false); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "option_unsupported") }
        XCTAssertEqual(h.supervisor.state, .stopped)
        XCTAssertEqual(h.startLog.count, 0)
        XCTAssertNil(h.supervisor.snapshot().operation)
        await h.cleanup()
    }
}

final class ForeignListener {
    private let descriptor: Int32
    init(port: Int) throws {
        descriptor = socket(AF_INET, SOCK_STREAM, 0)
        var yes: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        guard result == 0, listen(descriptor, 8) == 0 else { throw CocoaError(.fileWriteUnknown) }
    }
    func stillOpen() -> Bool { fcntl(descriptor, F_GETFD) != -1 }
    func close() { Darwin.close(descriptor) }
}

@MainActor
final class MenuStabilityTests: XCTestCase {
    /// The menu is built from the supervisor, which polls every tick. A property that announces a change on every
    /// poll, equal or not, makes every observer rebuild; with a menu open that closes its submenu.
    func testSteadyPollingAnnouncesNothing() async throws {
        let h = try Harness()
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        await h.expect { h.supervisor.drainSupported != nil && h.supervisor.webChatAvailable != nil }
        var announcements = 0
        let observer = h.supervisor.objectWillChange.sink { _ in announcements += 1 }
        try await Task.sleep(nanoseconds: 2_500_000_000)   // ~25 polls at 0.1 s
        observer.cancel()
        XCTAssertEqual(announcements, 0, "polling an unchanged Splash must announce no change")
        await h.cleanup()
    }

    func testARealChangeIsStillAnnounced() async throws {
        let h = try Harness()
        h.supervisor.begin()
        var announcements = 0
        let observer = h.supervisor.objectWillChange.sink { _ in announcements += 1 }
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        observer.cancel()
        XCTAssertGreaterThan(announcements, 0)
        await h.cleanup()
    }

    func testTheChatPageIsFoundAndTheSwitchTurnsItOff() async throws {
        let h = try Harness()
        h.supervisor.begin()
        try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        await h.expect { h.supervisor.webChatAvailable == true }
        let status = h.supervisor.snapshot()
        XCTAssertEqual(status.webChat.url, "http://127.0.0.1:\(h.port)/")
        XCTAssertEqual(WebChatAction.decide(status, hasSelectedModel: true), .open(URL(string: "http://127.0.0.1:\(h.port)/")!))
        // Turning the page off is a saved change that waits for a restart, then passes --no-webui.
        try h.store.updateSettings { $0.serveWebChat = false }
        XCTAssertTrue(h.supervisor.snapshot().pendingChanges.changes.joined().contains("chat page"))
        _ = try await h.supervisor.switchTo(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready && h.startLog.count == 2 }
        XCTAssertTrue((h.startLog.last?["argv"] as? [String] ?? []).contains("--no-webui"))
        await h.expect { h.supervisor.webChatAvailable == false }
        let action = WebChatAction.decide(h.supervisor.snapshot(), hasSelectedModel: true)
        XCTAssertFalse(action.isEnabled, "\(action)")
        await h.cleanup()
    }
}
