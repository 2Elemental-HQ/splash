import Foundation

/// A read-only view of the Hugging Face cache. Splash installs models from it,
/// so a file here is the closest local signal that a start needs no download.
public struct HFCache: Sendable {
    public let root: URL

    public init(root: URL = HFCache.defaultRoot()) { self.root = root }

    public static func defaultRoot(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let cache = environment["HF_HUB_CACHE"], !cache.isEmpty { return URL(fileURLWithPath: cache) }
        if let home = environment["HF_HOME"], !home.isEmpty { return URL(fileURLWithPath: home).appendingPathComponent("hub") }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
    }

    func repoDirectory(_ repo: String) -> URL {
        root.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
    }

    func snapshotDirectory(_ repo: String, commit: String) -> URL {
        repoDirectory(repo).appendingPathComponent("snapshots/\(commit)")
    }

    func commit(for repo: String, revision: String?) -> String? {
        let fm = FileManager.default
        let revision = revision ?? "main"
        if revision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil {
            return fm.fileExists(atPath: snapshotDirectory(repo, commit: revision).path) ? revision : nil
        }
        let ref = repoDirectory(repo).appendingPathComponent("refs/\(revision)")
        guard let text = try? String(contentsOf: ref, encoding: .utf8) else { return nil }
        let commit = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return fm.fileExists(atPath: snapshotDirectory(repo, commit: commit).path) ? commit : nil
    }

    func files(_ repo: String, commit: String) -> [String] {
        let base = snapshotDirectory(repo, commit: commit)
        return (try? FileManager.default.subpathsOfDirectory(atPath: base.path)) ?? []
    }

    /// The DFlash2 draft repository Splash pairs with a model family.
    /// Mirrors `install/families.py` of the Splash release this was written for;
    /// an unknown family reports nil and is never treated as downloaded.
    public static func draftRepository(for modelId: String) -> String? {
        let lower = modelId.lowercased()
        if lower.contains("qwen3.8-27b") { return "incoai/Qwen3.8-27B-DFlash2" }
        if lower.contains("qwen3.6-35b-a3b") { return "incoai/Qwen3.6-35B-A3B-DFlash2" }
        return nil
    }

    public struct Assessment: Equatable, Sendable {
        public var availability: Availability
        /// Repositories a start would download.
        public var missing: [String]
        public var note: String?
    }

    public func assess(modelId: String, revision: String?) -> Assessment {
        let parts = modelId.split(separator: ":", maxSplits: 1).map(String.init)
        let repo = parts[0]
        let variant = parts.count > 1 ? parts[1] : nil
        var missing: [String] = []
        var note: String?

        if let commit = commit(for: repo, revision: revision) {
            let names = files(repo, commit: commit)
            if let variant {
                if !names.contains(where: { $0.lowercased().hasSuffix(".gguf") && $0.lowercased().contains(variant.lowercased()) }) {
                    missing.append(repo)
                }
            } else if !names.contains("config.json") || !names.contains(where: { $0.hasSuffix(".safetensors") }) {
                missing.append(repo)
            }
        } else {
            missing.append(repo)
        }

        if let draft = Self.draftRepository(for: modelId) {
            if let commit = commit(for: draft, revision: nil),
               files(draft, commit: commit).contains("config.json") {
                // present
            } else {
                missing.append(draft)
            }
        } else {
            missing.append("(draft unknown)")
            note = "The matching draft model is unknown to this app, so a download cannot be ruled out."
        }
        return Assessment(availability: missing.isEmpty ? .local : .notLocal, missing: missing, note: note)
    }

    public struct Candidate: Equatable, Sendable {
        public var modelId: String
        public var revision: String
    }

    /// Target repositories Splash itself pinned (`refs/splash/<install>/<commit>`).
    public func splashPinnedModels() -> [Candidate] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: root.path) else { return [] }
        var result: [Candidate] = []
        for entry in entries.sorted() where entry.hasPrefix("models--") {
            let repo = entry.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
            if repo.lowercased().hasSuffix("dflash2") { continue }
            let pins = root.appendingPathComponent(entry).appendingPathComponent("refs/splash")
            guard let installs = try? fm.contentsOfDirectory(atPath: pins.path) else { continue }
            var seen = Set<String>()
            for install in installs {
                let commits = (try? fm.contentsOfDirectory(atPath: pins.appendingPathComponent(install).path)) ?? []
                for commit in commits where commit.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil
                    && fm.fileExists(atPath: snapshotDirectory(repo, commit: commit).path) && seen.insert(commit).inserted {
                    result.append(Candidate(modelId: repo, revision: commit))
                }
            }
        }
        return result
    }
}

/// Size of a would-be download, from the Hub's file listing. Network access
/// happens only when a person or an API client asks for it.
public enum DownloadEstimator {
    public struct Estimate: Codable, Equatable, Sendable {
        public var repository: String
        public var bytes: Int64?
        public var error: String?
    }

    public static func estimate(modelId: String, revision: String?, missing: [String]) async -> [Estimate] {
        var result: [Estimate] = []
        let parts = modelId.split(separator: ":", maxSplits: 1).map(String.init)
        for repo in missing where repo != "(draft unknown)" {
            let revisionPart = repo == parts[0] ? (revision ?? "main") : "main"
            let variant = repo == parts[0] && parts.count > 1 ? parts[1] : nil
            result.append(await size(repo: repo, revision: revisionPart, variant: variant))
        }
        return result
    }

    private static func size(repo: String, revision: String, variant: String?) async -> Estimate {
        guard let revisionPath = revision.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revisionPath)?recursive=true") else {
            return Estimate(repository: repo, bytes: nil, error: "bad repository name")
        }
        var request = URLRequest(url: url, timeoutInterval: 15)
        request.setValue("splash-manager", forHTTPHeaderField: "User-Agent")
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                return Estimate(repository: repo, bytes: nil, error: "the Hub did not list this repository")
            }
            var total: Int64 = 0
            for item in items where item["type"] as? String == "file" {
                let path = (item["path"] as? String) ?? ""
                if let variant, !(path.lowercased().hasSuffix(".gguf") && path.lowercased().contains(variant.lowercased())) { continue }
                total += (item["size"] as? NSNumber)?.int64Value ?? 0
            }
            return Estimate(repository: repo, bytes: total, error: nil)
        } catch {
            return Estimate(repository: repo, bytes: nil, error: "the Hub could not be reached")
        }
    }
}
