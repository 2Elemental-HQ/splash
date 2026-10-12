import Foundation

/// A read-only view of the Hugging Face cache. It answers "are the files that
/// Splash needs in the cache?". It does not prove that a start works: only a
/// start that reaches ready does, and the app records that separately.
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

    /// Any installed snapshot: the one a ref names, else one Splash pinned, else the newest.
    /// Used where no revision is requested; the start then runs `--offline` on what is installed.
    func installedCommit(for repo: String, revision: String?) -> String? {
        if let exact = commit(for: repo, revision: revision) { return exact }
        let fm = FileManager.default
        let pins = repoDirectory(repo).appendingPathComponent("refs/splash")
        for install in (try? fm.contentsOfDirectory(atPath: pins.path)) ?? [] {
            for name in (try? fm.contentsOfDirectory(atPath: pins.appendingPathComponent(install).path)) ?? []
            where name.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil
                && fm.fileExists(atPath: snapshotDirectory(repo, commit: name).path) { return name }
        }
        let snapshots = repoDirectory(repo).appendingPathComponent("snapshots")
        let names = ((try? fm.contentsOfDirectory(atPath: snapshots.path)) ?? []).sorted {
            let a = (try? fm.attributesOfItem(atPath: snapshots.appendingPathComponent($0).path)[.modificationDate] as? Date) ?? .distantPast
            let b = (try? fm.attributesOfItem(atPath: snapshots.appendingPathComponent($1).path)[.modificationDate] as? Date) ?? .distantPast
            return a > b
        }
        return names.first
    }

    func files(_ repo: String, commit: String) -> [String] {
        let base = snapshotDirectory(repo, commit: commit)
        return (try? FileManager.default.subpathsOfDirectory(atPath: base.path)) ?? []
    }

    /// The draft repository of the family whose name the model id contains.
    /// The family table comes from the installed Splash. The match by name is an
    /// inference: the engine itself decides the family from the model's config at start.
    public static func draftRepository(for modelId: String, families: [SplashFamily]?) -> String? {
        let lower = modelId.lowercased()
        return families?.first { lower.contains($0.name.lowercased()) }?.draftRepo
    }

    /// The two official packages have direct replacements. Community packages
    /// are identified by their manifest, never by a repository-name suffix.
    public func unsupportedPackageNote(modelId: String, revision: String?) -> String? {
        let official = [
            "incoai/Qwen3.8-27B-Splash": "mlx-community/Qwen3.8-27B-4bit",
            "incoai/Qwen3.6-35B-A3B-Splash": "mlx-community/Qwen3.6-35B-A3B-4bit",
        ]
        if let replacement = official[modelId] {
            return "Splash 1.3.1 no longer loads Splash packages. Add \(replacement) as a new configuration; its revision differs from this package's. Your existing configuration and cached files are kept."
        }
        guard let commit = installedCommit(for: modelId, revision: revision),
              let data = try? Data(contentsOf: snapshotDirectory(modelId, commit: commit).appendingPathComponent("manifest.json")),
              let manifest = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let format = manifest["format"] as? [String: Any],
              let name = format["name"] as? String,
              ["splash-packed-q4", "splash-packed-q4-moe"].contains(name) else { return nil }
        return "Splash 1.3.1 no longer loads this Splash package. Add the same fine-tune's MLX or GGUF release as a new configuration. Your existing configuration and cached files are kept."
    }

    /// Match upstream's root-only GGUF selection; vision projectors are not targets.
    static func ggufTargets(_ names: [String]) -> [String] {
        names.filter { !$0.contains("/") && $0.hasSuffix(".gguf") && !$0.lowercased().contains("mmproj") }.sorted()
    }

    static func ggufVariants(_ targets: [String]) -> [String] {
        let parts = targets.map { String($0.dropLast(5)).components(separatedBy: "-") }
        guard let first = parts.first, parts.count > 1 else { return [] }
        var shared = 0
        while shared < first.count && parts.allSatisfy({ $0.count > shared && $0[shared] == first[shared] }) { shared += 1 }
        return parts.map { $0.dropFirst(shared).joined(separator: "-") }
    }

    public enum Kind { case target, draft }

    /// Files missing or unreadable in one snapshot, empty when the set looks complete.
    /// A dangling link counts as missing: `fileExists` follows links to their blob.
    func problems(repo: String, commit: String, kind: Kind, gguf variant: String?) -> [String] {
        let fm = FileManager.default
        let base = snapshotDirectory(repo, commit: commit)
        func present(_ name: String) -> Bool { fm.fileExists(atPath: base.appendingPathComponent(name).path) }
        let names = files(repo, commit: commit)
        var problems: [String] = []
        let targets = Self.ggufTargets(names)
        if variant == nil && !names.contains(where: { $0.hasSuffix(".safetensors") }) && !targets.isEmpty {
            guard targets.count == 1 else { return ["select a GGUF variant: " + Self.ggufVariants(targets).joined(separator: ", ")] }
            return present(targets[0]) ? [] : ["\(targets[0]) is unreadable"]
        }
        if let variant {
            let variants = Self.ggufVariants(targets)
            let exact = zip(targets, variants).filter { $0.1.caseInsensitiveCompare(variant) == .orderedSame }.map { $0.0 }
            let selected = exact.isEmpty ? targets.filter { String($0.dropLast(5)).lowercased().hasSuffix("-" + variant.lowercased()) } : exact
            guard selected.count == 1 else { return ["no single root .gguf file for \(variant)"] }
            return present(selected[0]) ? [] : ["\(selected[0]) is unreadable"]
        }
        if !present("config.json") { problems.append("config.json") }
        if kind == .target, !present("tokenizer.json") { problems.append("tokenizer.json") }
        if let indexData = try? Data(contentsOf: base.appendingPathComponent("model.safetensors.index.json")),
           let index = (try? JSONSerialization.jsonObject(with: indexData)) as? [String: Any],
           let map = index["weight_map"] as? [String: String] {
            for shard in Set(map.values).sorted() where !present(shard) { problems.append(shard) }
        } else if !names.contains(where: { $0.hasSuffix(".safetensors") && present($0) }) {
            problems.append("weights (*.safetensors)")
        }
        return problems
    }

    public enum DraftSource: String, Sendable { case installedSplash = "installed_splash", unknown }

    public struct Assessment: Equatable, Sendable {
        public var availability: Availability
        /// Repositories a start would download (absent, or incomplete).
        public var missing: [String]
        /// Per repository, the files that are absent or unreadable.
        public var incomplete: [String: [String]]
        public var draftSource: DraftSource
        public var note: String?
    }

    public func assess(modelId: String, revision: String?, families: [SplashFamily]?) -> Assessment {
        let parts = modelId.split(separator: ":", maxSplits: 1).map(String.init)
        let repo = parts[0]
        let variant = parts.count > 1 ? parts[1] : nil
        var missing: [String] = []
        var incomplete: [String: [String]] = [:]
        var notes: [String] = []

        func check(_ repo: String, revision: String?, kind: Kind, variant: String?, anyInstalled: Bool) {
            let commit = anyInstalled ? installedCommit(for: repo, revision: revision) : commit(for: repo, revision: revision)
            guard let commit else { missing.append(repo); return }
            let found = problems(repo: repo, commit: commit, kind: kind, gguf: variant)
            if !found.isEmpty { missing.append(repo); incomplete[repo] = found }
        }
        if let note = unsupportedPackageNote(modelId: modelId, revision: revision) {
            return Assessment(availability: .notLocal, missing: [], incomplete: [:], draftSource: .unknown, note: note)
        }
        check(repo, revision: revision, kind: .target, variant: variant, anyInstalled: revision == nil)

        var source = DraftSource.unknown
        if let draft = Self.draftRepository(for: modelId, families: families) {
            source = .installedSplash
            check(draft, revision: nil, kind: .draft, variant: nil, anyInstalled: true)
        } else {
            missing.append("(draft unknown)")
            notes.append(families == nil
                ? "The installed Splash's family table could not be read, so the draft model is unknown and a download cannot be ruled out."
                : "No model family of the installed Splash matches this model id, so the draft model is unknown and a download cannot be ruled out.")
        }
        if !incomplete.isEmpty { notes.append("Incomplete in the cache: " + incomplete.map { "\($0.key) (\($0.value.joined(separator: ", ")))" }.sorted().joined(separator: "; ")) }
        return Assessment(availability: missing.isEmpty ? .local : .notLocal, missing: missing, incomplete: incomplete,
                          draftSource: source, note: notes.isEmpty ? nil : notes.joined(separator: " "))
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
                    let targets = Self.ggufTargets(files(repo, commit: commit))
                    let variants = Self.ggufVariants(targets)
                    if targets.count > 1 {
                        for variant in variants where !variant.isEmpty {
                            result.append(Candidate(modelId: repo + ":" + variant, revision: commit))
                        }
                    } else {
                        result.append(Candidate(modelId: repo, revision: commit))
                    }
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
