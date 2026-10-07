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
        XCTAssertEqual(cache.assess(modelId: "a/b", revision: commitB).availability, .notLocal)
        XCTAssertEqual(cache.assess(modelId: "a/b", revision: commitA).availability, .notLocal, "the draft is unknown, so a download cannot be ruled out")
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
        try make("mlx-community/Qwen3.8-27B-4bit", files: ["config.json", "m.safetensors"])
        try make("incoai/Qwen3.8-27B-DFlash2", files: ["config.json", "model.safetensors"])
        let cache = HFCache(root: dir)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA).availability, .local)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: nil).availability, .local)
        XCTAssertEqual(cache.assess(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitB).availability, .notLocal)
        XCTAssertEqual(cache.splashPinnedModels(), [HFCache.Candidate(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA)])
    }
}
