import Foundation

/// The JSON the management API returns. Encoded with snake_case keys.
public struct ManagerStatus: Codable, Equatable, Sendable {
    public struct Manager: Codable, Equatable, Sendable {
        public var version: String
        public var startedAt: Date
    }
    public struct SplashInfo: Codable, Equatable, Sendable {
        public var installed: Bool
        public var version: String?
    }
    public struct ConfigRef: Codable, Equatable, Sendable {
        public var id: String
        public var displayName: String
        public var modelId: String
        public var revision: String?
    }
    public struct Endpoint: Codable, Equatable, Sendable {
        public var bindHost: String
        public var port: Int
        public var exposure: Exposure
        public var requiresApiKey: Bool
        public var localUrl: String
        public var tailnetUrl: String?
        public var openaiBasePath: String
        /// Host names the last start told Splash to accept, besides loopback.
        public var allowedHosts: [String]
    }
    public struct Readiness: Codable, Equatable, Sendable {
        public var processAlive: Bool
        public var httpReady: Bool
        /// The intended model is listed by `/v1/models`.
        public var modelLoaded: Bool
    }
    public struct Activity: Codable, Equatable, Sendable {
        public var activeRequests: Int?
        public var idle: Bool?
        /// True only when the last status reading proved no call is running.
        public var switchSafe: Bool
        public var reason: String?
    }
    public struct Process: Codable, Equatable, Sendable {
        public var pid: Int32
        public var startedAt: Date
        public var uptimeSeconds: Int
        public var adopted: Bool
    }
    public struct Retry: Codable, Equatable, Sendable {
        public var attempt: Int
        public var maxAttempts: Int
        public var nextAt: Date
    }
    /// The configuration the running process was started with. Not the saved one.
    public struct Applied: Codable, Equatable, Sendable {
        public var configId: String
        public var modelId: String
        public var revision: String?
        public var options: ConfigView.Options
        public var port: Int
        public var exposure: Exposure
        public var allowedHosts: [String]
        public var offline: Bool
    }
    /// Saved changes that the running process does not have yet.
    public struct Pending: Codable, Equatable, Sendable {
        public var restartRequired: Bool
        public var changes: [String]
    }
    public struct Drain: Codable, Equatable, Sendable {
        /// Whether the running Splash drains on request. nil until its status was read.
        public var supported: Bool?
        public var draining: Bool
    }
    public struct Conflict: Codable, Equatable, Sendable {
        /// `external_splash`, `foreign_service` or `unreadable_service`.
        public var kind: String
        public var pid: Int?
        public var modelId: String?
        public var message: String
    }

    public var apiVersion = 1
    public var manager: Manager
    public var state: RunState
    public var ownership: Ownership
    public var detail: String?
    public var splash: SplashInfo
    public var config: ConfigRef?
    public var applied: Applied?
    public var pendingChanges: Pending
    public var drain: Drain
    /// The lifecycle operation in progress: start, stop, switch, retry or adopt.
    public var operation: String?
    public var loadedModelId: String?
    public var endpoint: Endpoint
    public var readiness: Readiness
    public var activity: Activity
    public var process: Process?
    public var lastError: ErrorRecord?
    public var retry: Retry?
    public var conflict: Conflict?
}

public struct ConfigView: Codable, Equatable, Sendable {
    public struct Options: Codable, Equatable, Sendable {
        public var languageOnly: Bool
        public var maxContext: String?
        public var maxMemory: String?
        public var idleRelease: String?
        public var reasoningDefault: String
        public var disableAne: Bool
    }
    public var id: String
    public var displayName: String
    public var modelId: String
    public var revision: String?
    public var availability: Availability
    public var selected: Bool
    public var active: Bool
    public var missing: [String]
    public var downloadRequired: Bool
    /// Files that are absent or unreadable in the cache, by repository.
    public var incomplete: [String: [String]]
    /// `installed_splash`: the draft comes from the installed Splash's family table; `unknown` otherwise.
    public var draftSource: String
    /// When a start of this model and revision last reached ready. nil: never proven here.
    public var verifiedStartAt: Date?
    public var note: String?
    public var options: Options
}

public struct SwitchCheck: Equatable, Sendable {
    public var safe: Bool
    public var reason: String?
    public var activity: SplashClient.Activity?
}
