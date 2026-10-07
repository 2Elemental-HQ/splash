import Foundation

/// Configs and settings, in one JSON file without secrets.
@MainActor
public final class ConfigStore: ObservableObject {
    @Published public private(set) var state: PersistedState
    public let paths: AppPaths

    public init(paths: AppPaths) {
        self.paths = paths
        try? paths.prepare()
        if let data = try? Data(contentsOf: paths.configFile),
           let decoded = try? JSONDecoder().decode(PersistedState.self, from: data) {
            state = decoded
        } else {
            state = PersistedState()
        }
    }

    public var configs: [ModelConfig] { state.configs }
    public var settings: ManagerSettings { state.settings }

    public func config(id: String) -> ModelConfig? { state.configs.first { $0.id == id } }

    public func updateSettings(_ change: (inout ManagerSettings) -> Void) throws {
        var next = state
        change(&next.settings)
        _ = try Validation.port(next.settings.inferencePort, field: "inference port")
        _ = try Validation.port(next.settings.managementPort, field: "management port")
        guard next.settings.inferencePort != next.settings.managementPort else {
            throw Validation.Failure(field: "management port", message: "must differ from the inference port")
        }
        guard next.settings.inferenceExposure != .tailnet else {
            throw Validation.Failure(field: "inference exposure", message: "use loopback or all interfaces")
        }
        guard next.settings.managementExposure != .allInterfaces else {
            throw Validation.Failure(field: "management exposure", message: "all interfaces is not available")
        }
        next.settings.allowedHosts = try next.settings.allowedHosts.map(Validation.hostName)
        try commit(next)
    }

    /// Adds or replaces a config.
    @discardableResult
    public func save(_ config: ModelConfig) throws -> ModelConfig {
        let valid = try config.validated()
        var next = state
        if let index = next.configs.firstIndex(where: { $0.id == valid.id }) {
            next.configs[index] = valid
        } else {
            next.configs.append(valid)
        }
        try commit(next)
        return valid
    }

    /// Adds the models that Splash itself installed. It reads the Hugging Face cache (the pins in `refs/splash`),
    /// saves what is new and reports what happened. It never downloads anything, and a second run adds nothing.
    /// It does not see models of other runtimes (LM Studio and the like); those are added by id.
    @discardableResult
    public func importSplashInstalledModels(cache: HFCache = HFCache()) -> ImportResult {
        var result = ImportResult()
        let found = cache.splashPinnedModels()
        result.found = found.count
        var ids = Set(state.configs.map(\.id))
        for candidate in found {
            if state.configs.contains(where: { $0.modelId == candidate.modelId && $0.revision == candidate.revision }) {
                result.alreadyConfigured += 1
                continue
            }
            let id = ModelConfig.makeId(modelId: candidate.modelId, revision: candidate.revision, existing: ids)
            let name = candidate.modelId.split(separator: "/").last.map(String.init) ?? candidate.modelId
            do {
                try save(ModelConfig(id: id, displayName: name, modelId: candidate.modelId, revision: candidate.revision))
                ids.insert(id)
                result.added.append(name)
            } catch {
                result.errors.append("\(candidate.modelId): \(error)")
            }
        }
        if state.settings.selectedConfigId == nil, let first = state.configs.first {
            do { try updateSettings { $0.selectedConfigId = first.id } } catch { result.errors.append("selection: \(error)") }
        }
        return result
    }

    public static func verificationKey(modelId: String, revision: String?) -> String { "\(modelId)@\(revision ?? "")" }

    public func verifiedStart(modelId: String, revision: String?) -> Date? {
        state.verifiedStarts[Self.verificationKey(modelId: modelId, revision: revision)]
    }

    public func markVerified(modelId: String, revision: String?) {
        var next = state
        next.verifiedStarts[Self.verificationKey(modelId: modelId, revision: revision)] = Date()
        try? commit(next)
    }

    public func remove(id: String) throws {
        var next = state
        next.configs.removeAll { $0.id == id }
        if next.settings.selectedConfigId == id { next.settings.selectedConfigId = next.configs.first?.id }
        try commit(next)
    }

    private func commit(_ next: PersistedState) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try paths.prepare()
        try Files.writeAtomically(try encoder.encode(next), to: paths.configFile)
        state = next
    }
}

public struct ImportResult: Equatable, Sendable {
    public var found = 0
    public var added: [String] = []
    public var alreadyConfigured = 0
    public var errors: [String] = []

    public init() {}

    /// What to tell a person.
    public var summary: String {
        var lines: [String] = []
        if found == 0 && errors.isEmpty {
            lines.append("No models that Splash installed were found in the Hugging Face cache. Models of other runtimes (such as LM Studio) are not detected; add them with Add model.")
        } else {
            lines.append("\(added.count) added, \(alreadyConfigured) already configured (\(found) found).")
            if !added.isEmpty { lines.append("Added: " + added.joined(separator: ", ") + ".") }
        }
        if !errors.isEmpty { lines.append("Problems: " + errors.joined(separator: "; ")) }
        return lines.joined(separator: " ")
    }
}
