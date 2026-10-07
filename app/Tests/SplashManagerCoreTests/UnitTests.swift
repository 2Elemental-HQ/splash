import XCTest
@testable import SplashManagerCore

final class ValidationTests: XCTestCase {
    func testModelIds() throws {
        XCTAssertEqual(try Validation.modelId("mlx-community/Qwen3.8-27B-4bit"), "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(try Validation.modelId("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"), "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M")
        for bad in ["", "no-slash", "a/b; rm -rf /", "--model/x", "a/b c", "a/b/c", "../x/y", "a/-b", "a/b:", "x/y\n--port"] {
            XCTAssertThrowsError(try Validation.modelId(bad), bad)
        }
    }
    func testRevisions() throws {
        XCTAssertNil(try Validation.revision(nil))
        XCTAssertNil(try Validation.revision("  "))
        XCTAssertEqual(try Validation.revision("10c35caafbb80f7dc6a7a432cdd11af10a6d4818"), "10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
        XCTAssertEqual(try Validation.revision("refs/pr/3"), "refs/pr/3")
        for bad in ["--offline", "-x", "a b", "a..b", "x;y"] { XCTAssertThrowsError(try Validation.revision(bad), bad) }
    }
    func testSizesAndDurations() throws {
        XCTAssertEqual(try Validation.size("28G", field: "x"), "28G")
        XCTAssertEqual(try Validation.size("auto", field: "x"), "auto")
        XCTAssertThrowsError(try Validation.size("28 G; ls", field: "x"))
        XCTAssertEqual(try Validation.duration("off", field: "x"), "off")
        XCTAssertEqual(try Validation.duration("1.5h", field: "x"), "1.5h")
        XCTAssertThrowsError(try Validation.duration("-5", field: "x"))
    }
    func testConfigIdsAreUnique() {
        let id = ModelConfig.makeId(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: "10c35caa", existing: ["qwen3-8-27b-4bit-10c35ca"])
        XCTAssertEqual(id, "qwen3-8-27b-4bit-10c35ca-2")
    }
}

final class HTTPParserTests: XCTestCase {
    func testParsesRequestWithBody() {
        let raw = "POST /api/v1/start?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\nAuthorization: Bearer t\r\n\r\n{}"
        guard case .request(let method, let target, let headers, let body) = HTTPParser.parse(Data(raw.utf8)) else { return XCTFail() }
        XCTAssertEqual(method, "POST"); XCTAssertEqual(target, "/api/v1/start?x=1")
        XCTAssertEqual(headers["authorization"], "Bearer t"); XCTAssertEqual(body, Data("{}".utf8))
        XCTAssertEqual(HTTPParser.split(target: target).query["x"], "1")
    }
    func testNeedsMoreUntilBodyComplete() {
        XCTAssertEqual(HTTPParser.parse(Data("POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nab".utf8)), .needMore)
        XCTAssertEqual(HTTPParser.parse(Data("GET / HTTP/1.1\r\nHost".utf8)), .needMore)
    }
    func testRejectsBadInput() {
        func status(_ s: String) -> Int? { if case .invalid(let c, _) = HTTPParser.parse(Data(s.utf8)) { return c }; return nil }
        XCTAssertEqual(status("GARBAGE\r\n\r\n"), 400)
        XCTAssertEqual(status("POST / HTTP/1.1\r\nContent-Length: 99999\r\n\r\n"), 413)
        XCTAssertEqual(status("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), 501)
        XCTAssertEqual(status("GET / HTTP/1.1\r\nA: 1\r\nA: 2\r\n\r\n"), 400)
        XCTAssertEqual(status("GET / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"), 400)
        XCTAssertEqual(HTTPParser.parse(Data(repeating: 0x41, count: 20_000)), .invalid(status: 431, message: "headers too large"))
    }
}

final class RedactorTests: XCTestCase {
    func testSecretsAreRemoved() {
        XCTAssertFalse(Redactor.redact("Authorization: Bearer abcdef123456789").contains("abcdef"))
        XCTAssertFalse(Redactor.redact("start --api-key sk-abcdefghijk123").contains("abcdefghijk"))
        XCTAssertFalse(Redactor.redact("token=hf_abcdefghijklmnop").contains("abcdefghij"))
        XCTAssertFalse(Redactor.redact("{\"api_key\": \"supersecretvalue\"}").contains("supersecretvalue"))
        XCTAssertEqual(Redactor.redact("10:30:54 Done · input 21 · output 400"), "10:30:54 Done · input 21 · output 400")
    }
}

final class ServeArgumentsTests: XCTestCase {
    let install = SplashInstall(executable: URL(fileURLWithPath: "/x"), version: "1",
                                serveFlags: ["--model", "--revision", "--port", "--host", "--offline", "--allowed-host",
                                             "--max-context", "--default-reasoning-effort"])
    func testBuildsOnlyKnownFlags() throws {
        var config = ModelConfig(id: "a", displayName: "A", modelId: "x/y", revision: "main")
        config.options.maxContext = "100K"; config.options.reasoning = .off
        var settings = ManagerSettings(); settings.allowedHosts = ["mac.ts.net"]
        let args = try ServeArguments.build(config: config, settings: settings, install: install, host: "127.0.0.1", offline: true)
        XCTAssertEqual(args, ["serve", "--model", "x/y", "--revision", "main", "--port", "8000", "--host", "127.0.0.1",
                              "--allowed-host", "mac.ts.net", "--offline", "--max-context", "100K",
                              "--default-reasoning-effort", "none"])
    }
    func testRejectsUnsupportedOption() {
        var config = ModelConfig(id: "a", displayName: "A", modelId: "x/y")
        config.options.disableANE = true
        XCTAssertThrowsError(try ServeArguments.build(config: config, settings: ManagerSettings(), install: install, host: "127.0.0.1", offline: false))
    }
    func testRejectsInjectedValues() {
        let config = ModelConfig(id: "a", displayName: "A", modelId: "x/y --api-key evil")
        XCTAssertThrowsError(try ServeArguments.build(config: config, settings: ManagerSettings(), install: install, host: "127.0.0.1", offline: false))
    }
}

final class HFCacheTests: XCTestCase {
    func testRevisionMustMatchAnInstalledSnapshot() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("hf-\(UUID().uuidString.prefix(6))")
        let snap = dir.appendingPathComponent("models--a--b/snapshots/\(commitA)")
        try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
        try Data().write(to: snap.appendingPathComponent("config.json"))
        try Data().write(to: snap.appendingPathComponent("w.safetensors"))
        let cache = HFCache(root: dir)
        XCTAssertEqual(cache.commit(for: "a/b", revision: commitA), commitA)
        XCTAssertNil(cache.commit(for: "a/b", revision: commitB))
        XCTAssertNil(cache.commit(for: "a/b", revision: nil), "no refs/main file")
        XCTAssertEqual(cache.assess(modelId: "a/b", revision: commitB, families: testFamilies).availability, .notLocal)
        XCTAssertEqual(cache.assess(modelId: "a/b", revision: commitA, families: testFamilies).availability, .notLocal, "the draft is unknown, so a download cannot be ruled out")
    }
}

final class HFCachePinnedTests: XCTestCase {
    /// The layout Splash itself leaves: pins under refs/splash, no refs/main.
    func testSplashPinnedInstallIsLocal() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("hf-\(UUID().uuidString.prefix(6))")
        func make(_ repo: String, files: [String]) throws {
            let base = dir.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            let snap = base.appendingPathComponent("snapshots/\(commitA)")
            let pin = base.appendingPathComponent("refs/splash/install1")
            try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: pin, withIntermediateDirectories: true)
            try Data().write(to: pin.appendingPathComponent(commitA))
            for f in files { try Data().write(to: snap.appendingPathComponent(f)) }
        }
        try make("mlx-community/Qwen3.8-27B-4bit", files: ["config.json", "tokenizer.json", "m.safetensors"])
        try make("incoai/Qwen3.8-27B-DFlash2", files: ["config.json", "model.safetensors"])
        let cache = HFCache(root: dir)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA, families: testFamilies).availability, .local)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: nil, families: testFamilies).availability, .local)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitB, families: testFamilies).availability, .notLocal)
        XCTAssertEqual(cache.splashPinnedModels(), [HFCache.Candidate(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA)])
    }
}

final class FamilyDiscoveryTests: XCTestCase {
    func testFamiliesAreReadFromTheInstalledSplash() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("fam-\(UUID().uuidString.prefix(6))")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("install"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("bin"), withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("install/__init__.py"))
        try """
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class ModelFamily:
            name: str
            draft_repo: str
        FAMILIES = (ModelFamily("Foo-1B", "acme/Foo-1B-Draft"),)
        """.write(to: root.appendingPathComponent("install/families.py"), atomically: true, encoding: .utf8)
        let executable = root.appendingPathComponent("bin/splash")
        try "#!/bin/sh\n".write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let families = await SplashFamilies.load(executable: executable)
        XCTAssertEqual(families, [SplashFamily(name: "Foo-1B", draftRepo: "acme/Foo-1B-Draft")])
        XCTAssertEqual(HFCache.draftRepository(for: "x/Foo-1B-4bit", families: families), "acme/Foo-1B-Draft")
        XCTAssertNil(HFCache.draftRepository(for: "x/Other-7B", families: families), "an unknown model stays unknown")
        XCTAssertNil(HFCache.draftRepository(for: "x/Foo-1B", families: nil))
    }

    func testNoFamilyTableMeansUnknown() async {
        let families = await SplashFamilies.load(executable: URL(fileURLWithPath: "/usr/bin/true"))
        XCTAssertNil(families)
    }

    func testHomebrewSplashFamiliesWhenInstalled() async throws {
        guard let url = SplashLocator.find(preferred: nil) else { throw XCTSkip("Splash is not installed") }
        let families = await SplashFamilies.load(executable: url)
        XCTAssertNotNil(families)
        XCTAssertTrue(families?.contains { $0.name == "Qwen3.8-27B" } ?? false)
    }
}

final class RequirementsTests: XCTestCase {
    func v(_ major: Int, _ minor: Int) -> OperatingSystemVersion { .init(majorVersion: major, minorVersion: minor, patchVersion: 0) }

    func testChipGeneration() {
        XCTAssertEqual(SystemRequirements.chipGeneration("Apple M4 Max"), 4)
        XCTAssertEqual(SystemRequirements.chipGeneration("Apple M1"), 1)
        XCTAssertEqual(SystemRequirements.chipGeneration("Apple M12 Ultra"), 12)
        XCTAssertNil(SystemRequirements.chipGeneration("Intel(R) Core(TM) i9"))
        XCTAssertNil(SystemRequirements.chipGeneration(nil))
    }

    func testSupportedMac() {
        let checks = SystemRequirements.evaluate(appleSilicon: true, chipBrand: "Apple M3 Pro", os: v(26, 4))
        XCTAssertTrue(checks.allSatisfy { $0.ok == true })
    }

    func testOldChipAndOldMacOSAreReportedSeparately() {
        let m1 = SystemRequirements.evaluate(appleSilicon: true, chipBrand: "Apple M1", os: v(26, 4))
        XCTAssertEqual(m1.first { $0.id == "chip" }?.ok, false)
        XCTAssertEqual(m1.first { $0.id == "os" }?.ok, true)
        let old = SystemRequirements.evaluate(appleSilicon: true, chipBrand: "Apple M4", os: v(26, 3))
        XCTAssertEqual(old.first { $0.id == "os" }?.ok, false)
        XCTAssertEqual(SystemRequirements.evaluate(appleSilicon: true, chipBrand: "Apple M4", os: v(27, 0)).first { $0.id == "os" }?.ok, true)
    }

    func testIntelIsNeverSupported() {
        let checks = SystemRequirements.evaluate(appleSilicon: false, chipBrand: "Intel(R) Core(TM) i7", os: v(26, 4))
        XCTAssertEqual(checks.first { $0.id == "arch" }?.ok, false)
        XCTAssertEqual(checks.first { $0.id == "chip" }?.ok, false)
    }
}

final class RuntimeBundleTests: XCTestCase {
    func stage(release: [String: Any]? = nil, extra: [String: String] = [:]) throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("rt-\(UUID().uuidString.prefix(6))")
        let fm = FileManager.default
        for dir in ["bin", "engine", "server"] { try fm.createDirectory(at: root.appendingPathComponent(dir), withIntermediateDirectories: true) }
        try "#!/bin/sh\necho Splash".write(to: root.appendingPathComponent("bin/splash"), atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: root.appendingPathComponent("bin/splash").path)
        try Data("engine".utf8).write(to: root.appendingPathComponent("engine/splash"))
        try Data("lib".utf8).write(to: root.appendingPathComponent("engine/splash.metallib"))
        try Data("server".utf8).write(to: root.appendingPathComponent("server/server.py"))
        let engine = RuntimeBundle.sha256(of: root.appendingPathComponent("engine/splash"))!
        let metallib = RuntimeBundle.sha256(of: root.appendingPathComponent("engine/splash.metallib"))!
        let rel = release ?? ["version": "9.9.9-test", "binary_sha256": engine, "metallib_sha256": metallib, "features": ["drain"]]
        try JSONSerialization.data(withJSONObject: rel).write(to: root.appendingPathComponent("release.json"))
        var files = ["engine/splash": engine, "engine/splash.metallib": metallib,
                     "server/server.py": RuntimeBundle.sha256(of: root.appendingPathComponent("server/server.py"))!,
                     "bin/splash": RuntimeBundle.sha256(of: root.appendingPathComponent("bin/splash"))!,
                     "release.json": RuntimeBundle.sha256(of: root.appendingPathComponent("release.json"))!]
        files.merge(extra) { $1 }
        try JSONSerialization.data(withJSONObject: ["runtime_version": rel["version"] ?? "", "files": files]).write(to: root.appendingPathComponent("runtime-manifest.json"))
        return root
    }

    func testIntactRuntimeVerifies() throws {
        let check = RuntimeBundle.verify(root: try stage())
        XCTAssertEqual(check.state, .verified, "\(check.problems)")
        XCTAssertEqual(check.version, "9.9.9-test")
        XCTAssertTrue(check.drainDeclared)
    }

    func testAChangedFileFailsTheCheck() throws {
        let root = try stage()
        try Data("tampered".utf8).write(to: root.appendingPathComponent("server/server.py"))
        let check = RuntimeBundle.verify(root: root)
        XCTAssertEqual(check.state, .failed)
        XCTAssertTrue(check.problems.contains("changed: server/server.py"))
    }

    func testAMissingFileAndAReplacedEngineFail() throws {
        let root = try stage()
        try FileManager.default.removeItem(at: root.appendingPathComponent("engine/splash.metallib"))
        XCTAssertTrue(RuntimeBundle.verify(root: root).problems.contains("missing or unreadable: engine/splash.metallib"))
    }

    func testReleaseJsonMustDescribeTheEngineThatShips() throws {
        let root = try stage(release: ["version": "9.9.9-test", "binary_sha256": String(repeating: "0", count: 64), "features": []])
        let check = RuntimeBundle.verify(root: root)
        XCTAssertEqual(check.state, .failed)
        XCTAssertTrue(check.problems.contains { $0.contains("binary_sha256") })
        XCTAssertFalse(check.drainDeclared)
    }

    func testManifestPathsCannotEscapeTheRuntime() throws {
        let check = RuntimeBundle.verify(root: try stage(extra: ["../../etc/hosts": "00"]))
        XCTAssertEqual(check.state, .failed)
        XCTAssertTrue(check.problems.contains { $0.hasPrefix("unexpected path") })
    }

    func testMissingManifestFails() throws {
        let root = try stage()
        try FileManager.default.removeItem(at: root.appendingPathComponent("runtime-manifest.json"))
        XCTAssertEqual(RuntimeBundle.verify(root: root).state, .failed)
    }

    func testBundledRuntimeIsPreferredAndInstalledOneIsLeftAlone() throws {
        let root = try stage()
        XCTAssertEqual(SplashLocator.resolve(preferred: nil, bundledRoot: root)?.source, .bundled)
        XCTAssertEqual(SplashLocator.resolve(preferred: nil, preferInstalled: true, bundledRoot: root)?.source,
                       FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/splash") || FileManager.default.isExecutableFile(atPath: "/usr/local/bin/splash") ? .installed : .bundled)
        XCTAssertEqual(SplashLocator.resolve(preferred: "/bin/ls", bundledRoot: root)?.source, .custom)
        XCTAssertNil(SplashLocator.resolve(preferred: nil, bundledRoot: nil).flatMap { $0.source == .bundled ? $0 : nil })
    }
}

@MainActor
final class StoredConfigurationTests: XCTestCase {
    func makeStore(_ json: String) throws -> ConfigStore {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("cfg-\(UUID().uuidString.prefix(6))")
        let paths = AppPaths(support: dir.appendingPathComponent("support"), logs: dir.appendingPathComponent("logs"))
        try paths.prepare()
        try json.write(to: paths.configFile, atomically: true, encoding: .utf8)
        return ConfigStore(paths: paths)
    }

    /// A configuration file as version 0.2.0 wrote it: no preferInstalledSplash, no serveWebChat, no verifiedStarts.
    let version02 = """
    {"configs":[{"id":"qwen","displayName":"Qwen","modelId":"mlx-community/Qwen3.8-27B-4bit","revision":"10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
                 "options":{"disableANE":false,"languageOnly":true,"reasoning":"off","maxContext":"64K"}}],
     "settings":{"allowedHosts":["mac.ts.net"],"autoRestart":false,"inferenceExposure":"all_interfaces","inferencePort":8123,
                 "managementEnabled":true,"managementExposure":"tailnet","managementPort":8800,"persistLogsToFile":true,
                 "selectedConfigId":"qwen","startSplashWhenAppLaunches":true,"stopSplashWhenAppQuits":false,"splashPath":"/opt/homebrew/bin/splash"}}
    """

    func testAnOlderFileKeepsEverythingItHad() throws {
        let store = try makeStore(version02)
        XCTAssertEqual(store.configs.count, 1, "the configuration must not be reset")
        XCTAssertEqual(store.configs[0].options.maxContext, "64K")
        XCTAssertTrue(store.configs[0].options.languageOnly)
        XCTAssertEqual(store.settings.inferencePort, 8123)
        XCTAssertEqual(store.settings.inferenceExposure, .allInterfaces)
        XCTAssertEqual(store.settings.allowedHosts, ["mac.ts.net"])
        XCTAssertEqual(store.settings.splashPath, "/opt/homebrew/bin/splash")
        XCTAssertFalse(store.settings.autoRestart)
        XCTAssertTrue(store.settings.startSplashWhenAppLaunches)
        // New keys take their defaults.
        XCTAssertFalse(store.settings.preferInstalledSplash)
        XCTAssertTrue(store.settings.serveWebChat)
    }

    func testAMinimalFileStillLoads() throws {
        let store = try makeStore(#"{"configs":[{"id":"a","modelId":"x/y"}],"settings":{}}"#)
        XCTAssertEqual(store.configs.first?.displayName, "a")
        XCTAssertEqual(store.settings.managementPort, 8765)
    }

    func testAnOldAppliedRecordStillDecodes() throws {
        let json = #"{"modelId":"x/y","options":{},"port":8000,"exposure":"loopback"}"#
        let spec = try JSONDecoder().decode(EffectiveSpec.self, from: Data(json.utf8))
        XCTAssertTrue(spec.serveWebChat)
        XCTAssertEqual(spec.allowedHosts, [])
    }

    // MARK: Import

    func makeCache(models: [(String, String)]) throws -> HFCache {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("hf-\(UUID().uuidString.prefix(6))")
        for (repo, commit) in models {
            let base = root.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            try FileManager.default.createDirectory(at: base.appendingPathComponent("snapshots/\(commit)"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: base.appendingPathComponent("refs/splash/install1"), withIntermediateDirectories: true)
            try Data().write(to: base.appendingPathComponent("refs/splash/install1/\(commit)"))
        }
        return HFCache(root: root)
    }

    func testImportReportsAndNeverDuplicates() throws {
        let store = try makeStore(#"{"configs":[],"settings":{}}"#)
        let cache = try makeCache(models: [("a/one", commitA), ("b/two", commitB)])
        let first = store.importSplashInstalledModels(cache: cache)
        XCTAssertEqual(first.found, 2); XCTAssertEqual(first.added.count, 2); XCTAssertEqual(first.alreadyConfigured, 0)
        XCTAssertEqual(store.configs.count, 2)
        XCTAssertEqual(store.settings.selectedConfigId, store.configs.first?.id)
        let again = store.importSplashInstalledModels(cache: cache)
        XCTAssertEqual(again.added.count, 0); XCTAssertEqual(again.alreadyConfigured, 2)
        XCTAssertEqual(store.configs.count, 2, "a second click adds nothing")
        XCTAssertTrue(again.summary.contains("0 added, 2 already configured"))
    }

    func testImportSaysWhenNothingIsFound() throws {
        let store = try makeStore(#"{"configs":[],"settings":{}}"#)
        let result = store.importSplashInstalledModels(cache: try makeCache(models: []))
        XCTAssertEqual(result.found, 0)
        XCTAssertTrue(result.summary.contains("No models that Splash installed were found"))
        XCTAssertTrue(result.summary.contains("LM Studio"), "it names what it does not see")
    }

    func testImportKeepsAConfigWithOtherOptionsAndAddsNoSecondCopy() throws {
        let store = try makeStore(#"{"configs":[],"settings":{}}"#)
        let cache = try makeCache(models: [("a/one", commitA)])
        store.importSplashInstalledModels(cache: cache)
        var config = store.configs[0]; config.options.maxContext = "32K"; try store.save(config)
        let result = store.importSplashInstalledModels(cache: cache)
        XCTAssertEqual(result.alreadyConfigured, 1)
        XCTAssertEqual(store.configs.count, 1)
        XCTAssertEqual(store.configs[0].options.maxContext, "32K")
    }

    func testImportReportsASaveError() throws {
        let store = try makeStore(#"{"configs":[],"settings":{}}"#)
        let cache = try makeCache(models: [("a/one", commitA)])
        // An immutable configuration file cannot be replaced: the save fails.
        var locked = URLResourceValues(); locked.isUserImmutable = true
        var file = store.paths.configFile
        try store.save(ModelConfig(id: "seed", displayName: "Seed", modelId: "x/seed"))
        try file.setResourceValues(locked)
        defer { var open = URLResourceValues(); open.isUserImmutable = false; try? file.setResourceValues(open) }
        let result = store.importSplashInstalledModels(cache: cache)
        XCTAssertEqual(result.added.count, 0)
        XCTAssertFalse(result.errors.isEmpty)
        XCTAssertTrue(result.summary.contains("Problems"))
    }
}

final class WebChatActionTests: XCTestCase {
    func status(state: RunState, ownership: Ownership = .managed, available: Bool? = nil, url: String? = nil) -> ManagerStatus {
        ManagerStatus(
            manager: .init(version: "t", startedAt: Date()), state: state, ownership: ownership, detail: nil,
            splash: .init(installed: true, version: "1", source: .bundled, integrity: "verified", drainDeclared: true, problem: nil),
            config: nil, applied: nil, pendingChanges: .init(restartRequired: false, changes: []),
            webChat: .init(available: available, url: url), drain: .init(supported: true, draining: false), operation: nil,
            loadedModelId: nil,
            endpoint: .init(bindHost: "127.0.0.1", port: 8000, exposure: .loopback, requiresApiKey: false, localUrl: "http://127.0.0.1:8000", tailnetUrl: nil, openaiBasePath: "/v1", allowedHosts: []),
            readiness: .init(processAlive: true, httpReady: true, modelLoaded: true),
            activity: .init(activeRequests: 0, idle: true, switchSafe: true, reason: nil),
            process: nil, lastError: nil, retry: nil, conflict: nil)
    }

    func testReadyWithTheChatPageOpensTheLocalAddress() {
        let action = WebChatAction.decide(status(state: .ready, available: true, url: "http://127.0.0.1:8123/"), hasSelectedModel: true)
        XCTAssertEqual(action, .open(URL(string: "http://127.0.0.1:8123/")!))
        XCTAssertFalse(URL(string: "http://127.0.0.1:8123/")!.absoluteString.contains("key"), "no key in the address")
    }

    func testStoppedOffersToStartFirst() {
        XCTAssertEqual(WebChatAction.decide(status(state: .stopped, ownership: .none), hasSelectedModel: true), .startThenOpen)
        XCTAssertEqual(WebChatAction.decide(status(state: .stopped, ownership: .none), hasSelectedModel: false), .unavailable("Add a model first."))
    }

    func testStartingAndStoppingWait() {
        XCTAssertFalse(WebChatAction.decide(status(state: .starting), hasSelectedModel: true).isEnabled)
        XCTAssertFalse(WebChatAction.decide(status(state: .stopping), hasSelectedModel: true).isEnabled)
    }

    func testAServerWithoutTheChatPageSaysSo() {
        let action = WebChatAction.decide(status(state: .ready, available: false, url: "http://127.0.0.1:8000/"), hasSelectedModel: true)
        guard case .unavailable(let why) = action else { return XCTFail("\(action)") }
        XCTAssertTrue(why.contains("chat page"))
        XCTAssertFalse(action.isEnabled)
    }

    func testReadyButNotYetChecked() {
        XCTAssertEqual(WebChatAction.decide(status(state: .ready, available: nil, url: "http://127.0.0.1:8000/"), hasSelectedModel: true).isEnabled, false)
    }

    func testFailedIsNotOpenable() {
        XCTAssertFalse(WebChatAction.decide(status(state: .failed, ownership: .none), hasSelectedModel: true).isEnabled)
    }
}
