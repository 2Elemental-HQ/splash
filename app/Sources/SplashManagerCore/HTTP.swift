import Foundation
import Network

public struct HTTPRequest: Sendable {
    public var method: String
    public var path: String
    public var query: [String: String]
    public var headers: [String: String]   // lowercased names
    public var body: Data
}

public struct HTTPResponse: Sendable {
    public var status: Int
    public var body: Data
    public var headers: [String: String] = [:]

    public static func json(_ status: Int, _ object: Any) -> HTTPResponse {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
        return HTTPResponse(status: status, body: data)
    }
}

/// A minimal HTTP/1.1 request parser. One request per connection.
public enum HTTPParser {
    public static let maxHeaderBytes = 16 * 1024
    public static let maxBodyBytes = 16 * 1024

    public enum Result: Equatable {
        case needMore
        case request(method: String, target: String, headers: [String: String], body: Data)
        case invalid(status: Int, message: String)
    }

    public static func parse(_ data: Data) -> Result {
        let separator = Data("\r\n\r\n".utf8)
        guard let end = data.range(of: separator) else {
            return data.count > maxHeaderBytes ? .invalid(status: 431, message: "headers too large") : .needMore
        }
        guard end.lowerBound <= maxHeaderBytes else { return .invalid(status: 431, message: "headers too large") }
        guard let head = String(data: data[data.startIndex..<end.lowerBound], encoding: .utf8) else {
            return .invalid(status: 400, message: "headers are not UTF-8")
        }
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1.") else {
            return .invalid(status: 400, message: "bad request line")
        }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { return .invalid(status: 400, message: "bad header") }
            let name = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            guard !name.isEmpty, headers[name] == nil else { return .invalid(status: 400, message: "bad or repeated header") }
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        if headers["transfer-encoding"] != nil { return .invalid(status: 501, message: "chunked bodies are not supported") }
        var length = 0
        if let value = headers["content-length"] {
            guard let parsed = Int(value), parsed >= 0 else { return .invalid(status: 400, message: "bad content length") }
            guard parsed <= maxBodyBytes else { return .invalid(status: 413, message: "body too large") }
            length = parsed
        }
        let bodyStart = end.upperBound
        guard data.count - bodyStart >= length else { return .needMore }
        return .request(method: requestLine[0], target: requestLine[1], headers: headers,
                        body: data[bodyStart..<(bodyStart + length)])
    }

    public static func split(target: String) -> (path: String, query: [String: String]) {
        guard let components = URLComponents(string: "http://x" + (target.hasPrefix("/") ? target : "/" + target)) else {
            return (target, [:])
        }
        var query: [String: String] = [:]
        for item in components.queryItems ?? [] { query[item.name] = item.value ?? "" }
        return (components.path, query)
    }

    static let reasons: [Int: String] = [
        200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
        409: "Conflict", 413: "Payload Too Large", 422: "Unprocessable Content", 424: "Failed Dependency",
        431: "Request Header Fields Too Large", 500: "Internal Server Error", 501: "Not Implemented", 503: "Service Unavailable",
    ]

    public static func serialize(_ response: HTTPResponse) -> Data {
        var head = "HTTP/1.1 \(response.status) \(reasons[response.status] ?? "Status")\r\n"
        head += "Content-Type: application/json\r\nContent-Length: \(response.body.count)\r\n"
        head += "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
        for (name, value) in response.headers { head += "\(name): \(value)\r\n" }
        head += "\r\n"
        return Data(head.utf8) + response.body
    }
}

public typealias HTTPHandler = @Sendable (HTTPRequest) async -> HTTPResponse

/// Serves HTTP on explicit addresses. It never binds the wildcard address.
public final class HTTPServer: @unchecked Sendable {
    public struct Binding: Equatable, Sendable { public var host: String; public var port: Int }

    private let queue = DispatchQueue(label: "splash-manager.http")
    private var listeners: [NWListener] = []
    private let handler: HTTPHandler
    public private(set) var boundTo: [Binding] = []
    public private(set) var failures: [String] = []

    public init(handler: @escaping HTTPHandler) { self.handler = handler }

    public func start(bindings: [Binding]) {
        stop()
        var started: [Binding] = []
        var failed: [String] = []
        for binding in bindings {
            guard let port = NWEndpoint.Port(rawValue: UInt16(binding.port)) else { continue }
            let parameters = NWParameters.tcp
            parameters.allowLocalEndpointReuse = true
            parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(binding.host), port: port)
            do {
                let listener = try NWListener(using: parameters)
                let ready = DispatchSemaphore(value: 0)
                var outcome: String?
                listener.stateUpdateHandler = { state in
                    switch state {
                    case .ready: ready.signal()
                    case .failed(let error): outcome = "\(binding.host):\(binding.port) \(error)"; ready.signal()
                    default: break
                    }
                }
                listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
                listener.start(queue: queue)
                if ready.wait(timeout: .now() + 3) == .timedOut { outcome = "\(binding.host):\(binding.port) timed out" }
                if let outcome { listener.cancel(); failed.append(outcome) } else { listeners.append(listener); started.append(binding) }
            } catch {
                failed.append("\(binding.host):\(binding.port) \(error)")
            }
        }
        boundTo = started
        failures = failed
    }

    public func stop() {
        listeners.forEach { $0.cancel() }
        listeners.removeAll()
        boundTo = []
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        let handler = self.handler
        let state = ConnectionState()
        queue.asyncAfter(deadline: .now() + 10) { if !state.finished { state.finished = true; connection.cancel() } }
        func receive() {
            connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { data, _, complete, error in
                if state.finished { return }
                if let data { state.buffer.append(data) }
                switch HTTPParser.parse(state.buffer) {
                case .needMore:
                    if error != nil || complete { state.finished = true; connection.cancel() } else { receive() }
                case .invalid(let status, let message):
                    state.finished = true
                    send(connection, HTTPResponse.json(status, ["error": ["code": "invalid_request", "message": message]]))
                case .request(let method, let target, let headers, let body):
                    state.finished = true
                    let (path, query) = HTTPParser.split(target: target)
                    let request = HTTPRequest(method: method, path: path, query: query, headers: headers, body: body)
                    Task { send(connection, await handler(request)) }
                }
            }
        }
        receive()
    }
}

private final class ConnectionState: @unchecked Sendable {
    var buffer = Data()
    var finished = false
}

private func send(_ connection: NWConnection, _ response: HTTPResponse) {
    connection.send(content: HTTPParser.serialize(response), completion: .contentProcessed { _ in connection.cancel() })
}
