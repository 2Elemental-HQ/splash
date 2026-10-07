import XCTest
@testable import SplashManagerCore

@MainActor
final class SupervisorTests: XCTestCase {
    func testStartReachesReadyOnlyWhenModelIsListed() async throws {
        let h = try Harness(control: ["ready_delay": 1.0])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        XCTAssertEqual(h.supervisor.state, .starting)
        // The process and its port exist, but /ready answers 503: not ready.
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertEqual(h.supervisor.state, .starting)
        let ready = await h.waitUntil { h.supervisor.state == .ready }
        XCTAssertTrue(ready)
        let status = h.supervisor.snapshot()
        XCTAssertTrue(status.readiness.httpReady && status.readiness.modelLoaded)
        XCTAssertEqual(status.loadedModelId, "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(status.ownership, .managed)
        // The model was local, so Splash was told not to ask the Hub.
        let argv = h.startLog.first?["argv"] as? [String] ?? []
        XCTAssertTrue(argv.contains("--offline"))
        XCTAssertEqual(argv[argv.firstIndex(of: "--revision")! + 1], commitA)
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
        do {
            _ = try await h.supervisor.start(configId: "b", allowDownload: true)
            XCTFail("expected refusal")
        } catch let e as AppError { XCTAssertEqual(e.code, "different_model_active") }
        XCTAssertEqual(h.startLog.count, 1)
        await h.cleanup()
    }

    func testDownloadNeedsConfirmation() async throws {
        let h = try Harness()
        h.supervisor.begin()
        do {
            _ = try await h.supervisor.start(configId: "b", allowDownload: false)
            XCTFail("expected refusal")
        } catch let e as AppError {
            XCTAssertEqual(e.code, "download_required")
            XCTAssertTrue(e.details?["missing"]?.contains("Qwen3.6-35B-A3B-4bit") == true)
        }
        XCTAssertEqual(h.startLog.count, 0)
        XCTAssertEqual(h.supervisor.state, .stopped)
        let views = h.supervisor.configViews()
        XCTAssertEqual(views.first { $0.id == "a" }?.availability, .local)
        XCTAssertEqual(views.first { $0.id == "b" }?.availability, .notLocal)
        await h.cleanup()
    }

    func testForeignListenerOnPortIsLeftAlone() async throws {
        let h = try Harness()
        h.supervisor.begin()
        let listener = try ForeignListener(port: h.port)
        defer { listener.close() }
        do {
            _ = try await h.supervisor.start(configId: "a", allowDownload: false)
            XCTFail("expected refusal")
        } catch let e as AppError { XCTAssertEqual(e.code, "port_in_use") }
        XCTAssertTrue(listener.stillOpen())
        XCTAssertEqual(h.startLog.count, 0)
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    func testExternalSplashIsRecognisedAndNeverStopped() async throws {
        let h = try Harness()
        h.supervisor.begin()
        // Start the fake outside the app.
        let external = Process()
        external.executableURL = h.fake
        external.arguments = ["serve", "--model", "mlx-community/Qwen3.8-27B-4bit", "--port", String(h.port)]
        external.standardOutput = FileHandle.nullDevice
        try external.run()
        defer { if external.isRunning { external.terminate() } }
        await h.expect { h.supervisor.ownership == .external && h.supervisor.state == .ready }
        XCTAssertEqual(h.supervisor.snapshot().conflict?.kind, "external_splash")
        do { _ = try await h.supervisor.stop(); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "not_managed") }
        do { _ = try await h.supervisor.start(configId: "b", allowDownload: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "external_instance") }
        // Same model: start is a no-op that reports the external instance.
        let same = try await h.supervisor.start(configId: "a", allowDownload: false)
        XCTAssertEqual(same.ownership, .external)
        XCTAssertTrue(external.isRunning)
        // When it goes away on its own, the app notices.
        external.terminate()
        await h.expect { h.supervisor.ownership == .none }
        await h.cleanup()
    }

    func testStopIsRefusedWhileACallRuns() async throws {
        let h = try Harness(control: ["hold": 3])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        async let call = h.chat()
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        do { _ = try await h.supervisor.stop(); XCTFail("expected refusal") }
        catch let e as AppError {
            XCTAssertEqual(e.code, "active_requests")
            XCTAssertEqual(e.details?["active_requests"], "1")
        }
        do { _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true); XCTFail("expected refusal") }
        catch let e as AppError { XCTAssertEqual(e.code, "active_requests") }
        XCTAssertEqual(h.supervisor.state, .ready, "the model was not touched")
        let code = await call
        XCTAssertEqual(code, 200, "the running call finished normally")
        // Idle now: stop works.
        _ = try await h.supervisor.stop()
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

    func testSwitchWhenIdleLoadsTheOtherModel() async throws {
        let h = try Harness()
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        let firstPid = h.supervisor.pid
        _ = try await h.supervisor.switchTo(configId: "b", allowDownload: true)
        await h.expect { h.supervisor.state == .ready && h.supervisor.loadedModelId == "mlx-community/Qwen3.6-35B-A3B-4bit" }
        XCTAssertNotEqual(firstPid, h.supervisor.pid)
        XCTAssertEqual(h.startLog.count, 2)
        await h.cleanup()
    }

    func testStopEscalatesWhenSigtermIsIgnored() async throws {
        var t = SupervisorTiming(); t.tick = 0.1; t.termGrace = 0.6; t.intGrace = 3; t.idleRecheck = 0.05
        let h = try Harness(control: ["ignore_term": true], timing: t)
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        let identity = Spawner.identity(of: h.supervisor.pid!)!
        _ = try await h.supervisor.stop()
        XCTAssertFalse(Spawner.isAlive(identity))
        XCTAssertEqual(h.supervisor.state, .stopped)
        await h.cleanup()
    }

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
        // Initial start plus three restarts, then it gives up.
        await h.expect(20) { h.startLog.count == 4 && h.supervisor.state == .failed && h.supervisor.retry == nil }
        try await Task.sleep(nanoseconds: 1_500_000_000)
        XCTAssertEqual(h.startLog.count, 4, "no unbounded restart loop")
        XCTAssertEqual(h.supervisor.lastError?.code, "crashed")
        await h.cleanup()
    }

    func testCrashThenRecovery() async throws {
        let h = try Harness(control: ["crash_after": 0.2, "crash_limit": 1])
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect(15) { h.startLog.count == 2 && h.supervisor.state == .ready }
        XCTAssertEqual(h.supervisor.ownership, .managed)
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

    func testAdoptionAfterAppRestartKeepsOwnership() async throws {
        let h = try Harness()
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        let pid = h.supervisor.pid!
        h.supervisor.shutdownSupervisor()
        // A new supervisor stands in for a restarted app.
        let second = SplashSupervisor(store: h.store, secrets: h.secrets, logs: LogBuffer(), timing: h.supervisor.timing)
        second.cache = h.supervisor.cache
        second.begin()
        await h.expect { second.state == .ready && second.ownership == .managed }
        XCTAssertEqual(second.pid, pid)
        XCTAssertTrue(second.adopted)
        _ = try await second.stop()
        _ = Spawner.reap(pid)   // in this test the runner is the parent; after a real app restart launchd is
        XCTAssertFalse(Spawner.isAlive(ProcessIdentity(pid: pid, startMicros: Spawner.startMicros(of: pid) ?? 0)))
        XCTAssertEqual(second.state, .stopped)
        second.shutdownSupervisor()
        try? FileManager.default.removeItem(at: h.dir)
    }

    func testStalePidRecordIsNotAdopted() async throws {
        let h = try Harness()
        // A record whose pid now belongs to an unrelated process (this test runner).
        let me = Spawner.identity(of: getpid())!
        let record = """
        {"identity":{"pid":\(me.pid),"startMicros":\(me.startMicros)},"configId":"a","modelId":"x/y","port":\(h.port),"exposure":"loopback"}
        """
        try h.paths.prepare()
        try record.write(to: h.paths.managedFile, atomically: true, encoding: .utf8)
        h.supervisor.begin()
        try await Task.sleep(nanoseconds: 800_000_000)
        XCTAssertEqual(h.supervisor.ownership, .none)
        XCTAssertFalse(FileManager.default.fileExists(atPath: h.paths.managedFile.path))
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
