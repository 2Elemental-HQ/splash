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
