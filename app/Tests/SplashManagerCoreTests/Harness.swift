import Foundation
import XCTest
@testable import SplashManagerCore

let commitA = String(repeating: "a", count: 40)
let commitB = String(repeating: "b", count: 40)

@MainActor
final class Harness {
    let dir: URL
    let fake: URL
    let store: ConfigStore
    let secrets = MemorySecretStore()
    let logs = LogBuffer()
    let supervisor: SplashSupervisor
    let port: Int
    let paths: AppPaths

    static func freePort() -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        _ = withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &length) } }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    init(control: [String: Any] = [:], timing: SupervisorTiming? = nil) throws {
        dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("sm-test-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/fake-splash")
        fake = dir.appendingPathComponent("fake-splash")
        try FileManager.default.copyItem(at: source, to: fake)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)
        try JSONSerialization.data(withJSONObject: control).write(to: dir.appendingPathComponent("control.json"))
        paths = AppPaths(support: dir.appendingPathComponent("support"), logs: dir.appendingPathComponent("logs"))
        store = ConfigStore(paths: paths)
        port = Harness.freePort()
        let fakePath = fake.path, inferencePort = port, managementPort = Harness.freePort()
        try store.updateSettings { $0.splashPath = fakePath; $0.inferencePort = inferencePort; $0.managementPort = managementPort }
        try store.save(ModelConfig(id: "a", displayName: "Model A", modelId: "mlx-community/Qwen3.8-27B-4bit", revision: commitA))
        try store.save(ModelConfig(id: "b", displayName: "Model B", modelId: "mlx-community/Qwen3.6-35B-A3B-4bit"))
        var t = timing ?? SupervisorTiming()
        if timing == nil {
            t.tick = 0.1; t.termGrace = 2; t.intGrace = 2; t.retryDelays = [0.3, 0.3, 0.3]; t.idleRecheck = 0.05
            t.notReadyTicksBeforeStarting = 3; t.startTimeoutLocal = 10
        }
        supervisor = SplashSupervisor(store: store, secrets: secrets, logs: logs, timing: t)
        try makeCache()
    }

    func makeCache() throws {
        let root = dir.appendingPathComponent("hf")
        func repo(_ name: String, commit: String, ref: String = "main", files: [String]) throws {
            let base = root.appendingPathComponent("models--" + name.replacingOccurrences(of: "/", with: "--"))
            let snap = base.appendingPathComponent("snapshots/\(commit)")
            try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: base.appendingPathComponent("refs"), withIntermediateDirectories: true)
            try commit.write(to: base.appendingPathComponent("refs/\(ref)"), atomically: true, encoding: .utf8)
            for f in files { try Data().write(to: snap.appendingPathComponent(f)) }
        }
        try repo("mlx-community/Qwen3.8-27B-4bit", commit: commitA, files: ["config.json", "model-1.safetensors"])
        try repo("incoai/Qwen3.8-27B-DFlash2", commit: commitB, files: ["config.json", "model.safetensors"])
        supervisor.cache = HFCache(root: root)
        // Model B has no files: starting it needs a download.
    }

    func setControl(_ control: [String: Any]) throws {
        try JSONSerialization.data(withJSONObject: control).write(to: dir.appendingPathComponent("control.json"))
    }

    var startLog: [[String: Any]] {
        guard let text = try? String(contentsOf: dir.appendingPathComponent("starts.log"), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] }
    }

    func waitUntil(_ timeout: TimeInterval = 15, _ condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return condition()
    }

    func expect(_ timeout: TimeInterval = 15, file: StaticString = #filePath, line: UInt = #line,
                _ condition: () -> Bool) async {
        let ok = await waitUntil(timeout, condition)
        XCTAssertTrue(ok, "condition not reached in \(timeout) s", file: file, line: line)
    }

    func chat(hold: TimeInterval? = nil) async -> Int {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = 30
        return ((try? await URLSession.shared.data(for: request).1) as? HTTPURLResponse)?.statusCode ?? 0
    }

    func cleanup() async {
        if supervisor.ownership == .managed { _ = try? await supervisor.stop(force: true) }
        supervisor.shutdownSupervisor()
        try? FileManager.default.removeItem(at: dir)
    }
}
