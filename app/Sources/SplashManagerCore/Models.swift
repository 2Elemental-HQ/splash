import Foundation

public enum ReasoningDefault: String, Codable, CaseIterable, Sendable {
    /// Leave the model's own template default untouched.
    case modelDefault = "model_default"
    /// `--default-reasoning-effort none`: Splash 1.3.0 turns thinking off for
    /// the tested Qwen models. Other efforts are template dependent and are
    /// not offered because their meaning is not established.
    case off
}

public struct ServeOptions: Codable, Equatable, Sendable {
    public var languageOnly = false
    public var maxContext: String?
    public var maxMemory: String?
    public var idleRelease: String?
    public var reasoning: ReasoningDefault = .modelDefault
    public var disableANE = false

    public init() {}

    public func validated() throws -> ServeOptions {
        var copy = self
        copy.maxContext = try Validation.size(maxContext, field: "max context")
        copy.maxMemory = try Validation.size(maxMemory, field: "max memory")
        copy.idleRelease = try Validation.duration(idleRelease, field: "idle release")
        return copy
    }
}

/// One model identity: a model id with an optional pinned revision, plus the
/// few serve options the app exposes. The id is the only handle the
/// management API accepts.
public struct ModelConfig: Codable, Identifiable, Equatable, Sendable {
    public var id: String
    public var displayName: String
    public var modelId: String
    public var revision: String?
    public var options: ServeOptions

    public init(id: String, displayName: String, modelId: String, revision: String? = nil,
                options: ServeOptions = ServeOptions()) {
        self.id = id
        self.displayName = displayName
        self.modelId = modelId
        self.revision = revision
        self.options = options
    }

    public func validated() throws -> ModelConfig {
        var copy = self
        copy.displayName = try Validation.configName(displayName)
        copy.modelId = try Validation.modelId(modelId)
        copy.revision = try Validation.revision(revision)
        copy.options = try options.validated()
        guard id.range(of: "^[a-z0-9][a-z0-9-]{0,63}$", options: .regularExpression) != nil else {
            throw Validation.Failure(field: "id", message: "use lowercase letters, digits and dashes")
        }
        return copy
    }

    /// A short id derived from the model, unique among `existing`.
    public static func makeId(modelId: String, revision: String?, existing: Set<String>) -> String {
        var base = modelId.split(separator: "/").last.map(String.init) ?? modelId
        base = base.lowercased().map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
        while base.contains("--") { base = base.replacingOccurrences(of: "--", with: "-") }
        base = String(base.trimmingCharacters(in: CharacterSet(charactersIn: "-")).prefix(48))
        if base.isEmpty { base = "model" }
        if let revision, !revision.isEmpty { base += "-" + String(revision.lowercased().filter { $0.isLetter || $0.isNumber }.prefix(7)) }
        var candidate = base, n = 2
        while existing.contains(candidate) { candidate = "\(base)-\(n)"; n += 1 }
        return candidate
    }
}

public enum Exposure: String, Codable, CaseIterable, Sendable {
    /// 127.0.0.1 only.
    case loopback
    /// The Tailscale address of this Mac (100.64.0.0/10) in addition to
    /// loopback where the component allows it.
    case tailnet
    /// 0.0.0.0. Inference only; the management API has no such mode.
    case allInterfaces = "all_interfaces"
}

public struct ManagerSettings: Codable, Equatable, Sendable {
    public var splashPath: String?
    public var inferencePort = 8000
    public var inferenceExposure: Exposure = .loopback
    public var allowedHosts: [String] = []
    public var managementEnabled = true
    public var managementPort = 8765
    /// `.loopback` or `.tailnet` only.
    public var managementExposure: Exposure = .loopback
    public var selectedConfigId: String?
    public var startSplashWhenAppLaunches = false
    public var stopSplashWhenAppQuits = true
    public var autoRestart = true
    public var persistLogsToFile = false

    public init() {}
}

public struct PersistedState: Codable, Equatable, Sendable {
    public var configs: [ModelConfig] = []
    public var settings = ManagerSettings()
    public init() {}
}

// MARK: - Runtime state

public enum RunState: String, Codable, Sendable {
    case stopped, starting, ready, stopping, failed
}

public enum Ownership: String, Codable, Sendable {
    /// Nothing runs.
    case none
    /// Started by this app, or adopted after an app restart with a verified identity.
    case managed
    /// A Splash that runs on the port and this app did not start. Never stopped here.
    case external
}

public enum Availability: String, Codable, Sendable {
    /// The configured model is the one loaded now.
    case loaded
    /// The files are in the local Hugging Face cache.
    case local
    /// A start would download files.
    case notLocal = "not_local"
}

public struct AppError: Error, Codable, Equatable, Sendable {
    public var code: String
    public var message: String
    public var httpStatus: Int
    public var details: [String: String]?

    public init(_ code: String, _ message: String, status: Int = 409, details: [String: String]? = nil) {
        self.code = code
        self.message = message
        self.httpStatus = status
        self.details = details
    }
}

public struct ErrorRecord: Codable, Equatable, Sendable {
    public var code: String
    public var message: String
    public var at: Date
}
