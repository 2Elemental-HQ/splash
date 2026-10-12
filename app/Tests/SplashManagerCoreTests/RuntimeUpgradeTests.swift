import XCTest
@testable import SplashManagerCore

final class RuntimeUpgradeTests: XCTestCase {
    func cache(_ files: [String: String], repo: String = "owner/Qwen3.8-27B-GGUF") throws -> HFCache {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let base = root.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
        let snapshot = base.appendingPathComponent("snapshots/\(commitA)")
        let pin = base.appendingPathComponent("refs/splash/install1")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pin, withIntermediateDirectories: true)
        try Data(commitA.utf8).write(to: pin.appendingPathComponent(commitA))
        for (name, content) in files {
            let file = snapshot.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(content.utf8).write(to: file)
        }
        return HFCache(root: root)
    }

    func testGGUFImportKeepsVariantsAndExcludesProjectorsAndSubfolders() throws {
        let c = try cache(["Model-Q4_K_M.gguf": "", "Model-UD-Q4_K_M.gguf": "",
                           "mmproj-F16.gguf": "", "split/Model-Q8_0.gguf": ""])
        XCTAssertEqual(c.splashPinnedModels().map(\.modelId), [
            "owner/Qwen3.8-27B-GGUF:Q4_K_M", "owner/Qwen3.8-27B-GGUF:UD-Q4_K_M",
        ])
        XCTAssertTrue(c.problems(repo: "owner/Qwen3.8-27B-GGUF", commit: commitA, kind: .target, gguf: "Q4_K_M").isEmpty)
        XCTAssertFalse(c.problems(repo: "owner/Qwen3.8-27B-GGUF", commit: commitA, kind: .target, gguf: nil).isEmpty)
        XCTAssertFalse(c.problems(repo: "owner/Qwen3.8-27B-GGUF", commit: commitA, kind: .target, gguf: "F16").isEmpty)
    }

    func testSingleGGUFNeedsNeitherVariantNorMLXMetadata() throws {
        let c = try cache(["Model-Q4_K_M.gguf": "", "mmproj-F16.gguf": ""])
        XCTAssertEqual(c.splashPinnedModels().map(\.modelId), ["owner/Qwen3.8-27B-GGUF"])
        XCTAssertTrue(c.problems(repo: "owner/Qwen3.8-27B-GGUF", commit: commitA, kind: .target, gguf: nil).isEmpty)
    }

    func testOfficialAndCommunityPackagesExplainMigrationWithoutChangingFiles() throws {
        let c = try cache(["manifest.json": #"{"format":{"name":"splash-packed-q4"}}"#], repo: "owner/finetune")
        XCTAssertTrue(c.unsupportedPackageNote(modelId: "incoai/Qwen3.8-27B-Splash", revision: commitA)!.contains("mlx-community/Qwen3.8-27B-4bit"))
        XCTAssertTrue(c.unsupportedPackageNote(modelId: "owner/finetune", revision: commitA)!.contains("same fine-tune"))
        XCTAssertNotNil(c.commit(for: "owner/finetune", revision: commitA))
        XCTAssertNil(c.unsupportedPackageNote(modelId: "mlx-community/Qwen3.8-27B-4bit", revision: nil))
    }

    @MainActor
    func testImportReportsLegacyPackageAndKeepsSavedConfiguration() throws {
        let repo = "incoai/Qwen3.8-27B-Splash"
        let c = try cache(["manifest.json": #"{"format":{"name":"splash-packed-q4"}}"#], repo: repo)
        let store = ConfigStore(paths: AppPaths(support: c.root.appendingPathComponent("settings"), logs: c.root.appendingPathComponent("logs")))
        try store.save(ModelConfig(id: "old", displayName: "My old model", modelId: repo, revision: commitA))
        let result = store.importSplashInstalledModels(cache: c)
        XCTAssertEqual(result.added.count, 0)
        XCTAssertEqual(result.errors.count, 1)
        XCTAssertEqual(store.config(id: "old")?.revision, commitA)
        XCTAssertEqual(store.config(id: "old")?.modelId, repo)
    }
    @MainActor
    func testLegacyPackageStartIsRefusedBeforeDownloadOrSpawn() async throws {
        let h = try Harness()
        try h.store.save(ModelConfig(id: "legacy", displayName: "Legacy", modelId: "incoai/Qwen3.8-27B-Splash", revision: commitA))
        h.supervisor.begin()
        do {
            _ = try await h.supervisor.start(configId: "legacy", allowDownload: true)
            XCTFail("a removed package must not start")
        } catch let error as AppError {
            XCTAssertEqual(error.code, "unsupported_model_package")
        }
        XCTAssertTrue(h.startLog.isEmpty)
        XCTAssertEqual(h.store.config(id: "legacy")?.revision, commitA)
        await h.cleanup()
    }

    func testThermalStatusDoesNotChangeActivityInterpretation() {
        let status: [String: Any] = [
            "http": ["requests": ["active": 1], "draining": false],
            "scheduler": ["decoding": 1, "queued": 0],
            "admission": ["waiting": 0],
            "thermal_state": "nominal",
        ]
        XCTAssertEqual(SplashClient.activity(from: status)?.activeRequests, 1)
        XCTAssertEqual(SplashClient.activity(from: status)?.generating, 1)
        XCTAssertEqual(SplashClient.activity(from: status)?.idle, false)
    }

    @MainActor
    func testSwitchToLegacyPackageLeavesTheRunningModelUntouched() async throws {
        let h = try Harness()
        h.supervisor.begin()
        _ = try await h.supervisor.start(configId: "a", allowDownload: false)
        await h.expect { h.supervisor.state == .ready }
        try h.store.save(ModelConfig(id: "legacy", displayName: "Legacy", modelId: "incoai/Qwen3.8-27B-Splash"))
        for allowDownload in [false, true] {
            do {
                _ = try await h.supervisor.switchTo(configId: "legacy", allowDownload: allowDownload)
                XCTFail("a removed package must not stop the running model")
            } catch let error as AppError {
                XCTAssertEqual(error.code, "unsupported_model_package")
            }
        }
        XCTAssertEqual(h.supervisor.state, .ready)
        XCTAssertEqual(h.supervisor.activeConfig?.id, "a")
        XCTAssertEqual(h.startLog.count, 1)
        XCTAssertFalse(h.supervisor.configViews().first { $0.id == "legacy" }!.downloadRequired)
        await h.cleanup()
    }

}
