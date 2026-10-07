import Foundation

/// Routes of the management API. Version 1, JSON, Bearer authentication.
///
/// The API takes configuration ids that a person saved in the app. It has no
/// route that accepts a command, an argument or a model id.
@MainActor
public final class ManagementAPI {
    private let supervisor: SplashSupervisor
    private let secrets: SecretStore
    private var failedAuth = 0

    public init(supervisor: SplashSupervisor, secrets: SecretStore) {
        self.supervisor = supervisor
        self.secrets = secrets
    }

    public nonisolated var handler: HTTPHandler {
        { [self] request in await self.handle(request) }
    }

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    func handle(_ request: HTTPRequest) async -> HTTPResponse {
        guard authorized(request) else {
            failedAuth += 1
            try? await Task.sleep(nanoseconds: 300_000_000)   // slows guessing
            var response = Self.error(AppError("unauthorized", "A valid management token is required.", status: 401))
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        }
        do {
            return try await route(request)
        } catch let error as AppError {
            return Self.error(error)
        } catch {
            return Self.error(AppError("internal_error", "Unexpected error.", status: 500))
        }
    }

    private func authorized(_ request: HTTPRequest) -> Bool {
        guard let header = request.headers["authorization"], header.lowercased().hasPrefix("bearer "),
              let token = secrets.read(SecretAccount.managementToken), !token.isEmpty else { return false }
        return Secrets.equal(String(header.dropFirst(7)).trimmingCharacters(in: .whitespaces), token)
    }

    private func route(_ request: HTTPRequest) async throws -> HTTPResponse {
        let parts = request.path.split(separator: "/").map(String.init)
        guard parts.count >= 3, parts[0] == "api", parts[1] == "v1" else { throw AppError("not_found", "Unknown route.", status: 404) }
        let tail = Array(parts.dropFirst(2))
        if request.method == "GET", tail.count == 3, tail[0] == "configs", tail[2] == "download-estimate" {
            let id = tail[1]
            guard let config = supervisor.store.config(id: id) else { throw AppError("unknown_config", "No such configuration.", status: 404) }
            let assessment = supervisor.assess(config)
            let estimates = await DownloadEstimator.estimate(modelId: config.modelId, revision: config.revision, missing: assessment.missing)
            return try encode(200, EstimateBody(configId: id, downloadRequired: assessment.availability == .notLocal,
                                                missing: assessment.missing, estimates: estimates))
        }
        switch (request.method, tail) {
        case ("GET", ["status"]):
            if let target = request.query["wait_for"] {
                guard let state = RunState(rawValue: target) else { throw AppError("invalid_request", "wait_for must be a state name.", status: 400) }
                let timeout = Double(request.query["timeout"] ?? "30") ?? 30
                return try encode(200, await supervisor.wait(for: state, timeout: timeout))
            }
            return try encode(200, supervisor.snapshot())
        case ("GET", ["configs"]):
            return try encode(200, ["configs": supervisor.configViews()])
        case ("GET", ["logs"]):
            let count = min(max(Int(request.query["lines"] ?? "100") ?? 100, 1), 500)
            let lines = supervisor.logs.lines.suffix(count).map { ["time": ISO8601DateFormatter().string(from: $0.date), "source": $0.source.rawValue, "text": $0.text] }
            return HTTPResponse.json(200, ["lines": lines])
        case ("POST", ["start"]):
            let body = try Self.body(request, allowed: ["config_id", "allow_download", "wait_ready_seconds"])
            let id = try Self.string(body, "config_id", required: true)!
            var status = try await supervisor.start(configId: id, allowDownload: Self.bool(body, "allow_download"))
            status = await waitIfAsked(body, status)
            return try encode(status.state == .ready ? 200 : 202, status)
        case ("POST", ["switch"]):
            let body = try Self.body(request, allowed: ["config_id", "allow_download", "wait_ready_seconds"])
            let id = try Self.string(body, "config_id", required: true)!
            var status = try await supervisor.switchTo(configId: id, allowDownload: Self.bool(body, "allow_download"))
            status = await waitIfAsked(body, status)
            return try encode(status.state == .ready ? 200 : 202, status)
        case ("POST", ["stop"]):
            let body = try Self.body(request, allowed: ["wait_stopped_seconds"])
            var status = try await supervisor.stop(force: false)
            if let seconds = (body["wait_stopped_seconds"] as? NSNumber)?.doubleValue, seconds > 0, status.state == .stopping {
                status = await supervisor.wait(for: .stopped, timeout: seconds)
            }
            return try encode(status.state == .stopping ? 202 : 200, status)
        case (_, ["status"]), (_, ["configs"]), (_, ["logs"]), (_, ["start"]), (_, ["switch"]), (_, ["stop"]):
            throw AppError("method_not_allowed", "Method not allowed for this route.", status: 405)
        default:
            throw AppError("not_found", "Unknown route.", status: 404)
        }
    }

    private func waitIfAsked(_ body: [String: Any], _ status: ManagerStatus) async -> ManagerStatus {
        guard let seconds = (body["wait_ready_seconds"] as? NSNumber)?.doubleValue, seconds > 0, status.state != .ready else { return status }
        return await supervisor.wait(for: .ready, timeout: seconds)
    }

    // MARK: Bodies

    static func body(_ request: HTTPRequest, allowed: Set<String>) throws -> [String: Any] {
        if request.body.isEmpty { return [:] }
        guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true else {
            throw AppError("invalid_request", "Send application/json.", status: 400)
        }
        guard let object = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
            throw AppError("invalid_request", "The body must be a JSON object.", status: 400)
        }
        let extra = Set(object.keys).subtracting(allowed)
        guard extra.isEmpty else {
            throw AppError("invalid_request", "Unknown field(s): \(extra.sorted().joined(separator: ", ")). This API accepts configuration ids only.", status: 400)
        }
        for key in ["wait_ready_seconds", "wait_stopped_seconds"] {
            if let wait = object[key] {
                guard let number = wait as? NSNumber, number.doubleValue >= 0, number.doubleValue <= 600 else {
                    throw AppError("invalid_request", "\(key) must be a number from 0 to 600.", status: 400)
                }
            }
        }
        return object
    }

    static func string(_ body: [String: Any], _ key: String, required: Bool) throws -> String? {
        guard let value = body[key] else {
            if required { throw AppError("invalid_request", "\(key) is required.", status: 400) }
            return nil
        }
        guard let text = value as? String, !text.isEmpty, text.count <= 64 else {
            throw AppError("invalid_request", "\(key) must be a configuration id.", status: 400)
        }
        return text
    }

    static func bool(_ body: [String: Any], _ key: String) throws -> Bool {
        guard let value = body[key] else { return false }
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw AppError("invalid_request", "\(key) must be true or false.", status: 400)
        }
        return number.boolValue
    }

    private struct EstimateBody: Codable {
        var configId: String
        var downloadRequired: Bool
        var missing: [String]
        var estimates: [DownloadEstimator.Estimate]
    }

    private func encode<T: Encodable>(_ status: Int, _ value: T) throws -> HTTPResponse {
        HTTPResponse(status: status, body: try Self.encoder.encode(value))
    }

    static func error(_ error: AppError) -> HTTPResponse {
        var inner: [String: Any] = ["code": error.code, "message": error.message]
        if let details = error.details { inner["details"] = details }
        return HTTPResponse.json(error.httpStatus, ["error": inner])
    }
}

/// Starts and stops the HTTP listeners that serve the management API.
@MainActor
public final class ManagementService: ObservableObject {
    @Published public private(set) var listening: [String] = []
    @Published public private(set) var problem: String?
    private let server: HTTPServer
    private let store: ConfigStore
    private var watcher: Task<Void, Never>?
    private var lastBindings: [HTTPServer.Binding] = []

    public init(api: ManagementAPI, store: ConfigStore) {
        self.server = HTTPServer(handler: api.handler)
        self.store = store
    }

    public func begin() {
        watcher = Task { [weak self] in
            while !Task.isCancelled {
                self?.reconcile()
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    /// Binds again when the settings or the Tailscale address change.
    public func reconcile() {
        let settings = store.settings
        var wanted: [HTTPServer.Binding] = []
        if settings.managementEnabled {
            wanted.append(.init(host: "127.0.0.1", port: settings.managementPort))
            if settings.managementExposure == .tailnet, let address = NetInfo.tailnetAddress() {
                wanted.append(.init(host: address, port: settings.managementPort))
            }
        }
        guard wanted != lastBindings else { return }
        lastBindings = wanted
        server.stop()
        if !wanted.isEmpty { server.start(bindings: wanted) }
        listening = server.boundTo.map { "\($0.host):\($0.port)" }
        var problems = server.failures
        if settings.managementEnabled, settings.managementExposure == .tailnet, NetInfo.tailnetAddress() == nil {
            problems.append("No Tailscale address; listening on loopback only")
        }
        problem = problems.isEmpty ? nil : problems.joined(separator: "; ")
        if !problems.isEmpty { lastBindings = [] }   // retry on the next pass
    }

    public func stop() { watcher?.cancel(); server.stop(); listening = [] }
}
