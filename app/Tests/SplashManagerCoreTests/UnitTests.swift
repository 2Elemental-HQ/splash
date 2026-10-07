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
