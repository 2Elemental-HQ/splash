import Foundation
import Darwin

public enum NetInfo {
    /// The IPv4 address Tailscale gave this Mac (100.64.0.0/10), if any.
    public static func tailnetAddress() -> String? {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return nil }
        defer { freeifaddrs(list) }
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let entry = cursor {
            defer { cursor = entry.pointee.ifa_next }
            guard let address = entry.pointee.ifa_addr, address.pointee.sa_family == UInt8(AF_INET) else { continue }
            let name = String(cString: entry.pointee.ifa_name)
            guard name.hasPrefix("utun") else { continue }
            var sin = sockaddr_in()
            memcpy(&sin, address, MemoryLayout<sockaddr_in>.size)
            let value = UInt32(bigEndian: sin.sin_addr.s_addr)
            if value & 0xFFC0_0000 == 0x6440_0000 {
                var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
                var raw = sin.sin_addr
                inet_ntop(AF_INET, &raw, &buffer, socklen_t(INET_ADDRSTRLEN))
                return String(cString: buffer)
            }
        }
        return nil
    }

    /// The MagicDNS name of this Mac (without the trailing dot), from the Tailscale CLI, if it can be read.
    public static func tailnetHostName() async -> String? {
        let candidates = ["/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale",
                          "/Applications/Tailscale.app/Contents/MacOS/Tailscale"]
        guard let path = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }),
              let text = await SplashLocator.run(URL(fileURLWithPath: path), ["status", "--json", "--peers=false"], timeout: 4),
              let json = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any],
              let me = json["Self"] as? [String: Any], var name = me["DNSName"] as? String else { return nil }
        if name.hasSuffix(".") { name.removeLast() }
        return (try? Validation.hostName(name)) ?? nil
    }

    /// True when something accepts TCP connections at host:port.
    public static func isListening(host: String, port: Int, timeout: TimeInterval = 1) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async { continuation.resume(returning: connect(host: host, port: port, timeout: timeout)) }
        }
    }

    static func connect(host: String, port: Int, timeout: TimeInterval) -> Bool {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else { return false }
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        let flags = fcntl(descriptor, F_GETFL)
        _ = fcntl(descriptor, F_SETFL, flags | O_NONBLOCK)
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if result == 0 { return true }
        guard errno == EINPROGRESS else { return false }
        var poller = pollfd(fd: descriptor, events: Int16(POLLOUT), revents: 0)
        guard poll(&poller, 1, Int32(timeout * 1000)) > 0 else { return false }
        var error: Int32 = 0
        var length = socklen_t(MemoryLayout<Int32>.size)
        getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &error, &length)
        return error == 0
    }
}

/// A small HTTP client for the one Splash instance this app watches.
public struct SplashClient: Sendable {
    public let host: String
    public let port: Int
    public let apiKey: String?
    private let session: URLSession

    private static let sharedSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 3
        configuration.timeoutIntervalForResource = 4
        configuration.connectionProxyDictionary = [:]
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        return URLSession(configuration: configuration)
    }()

    public init(host: String, port: Int, apiKey: String?) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
        session = Self.sharedSession
    }

    public func get(_ path: String, authenticated: Bool = true) async -> (status: Int, json: [String: Any]?) {
        guard let url = URL(string: "http://\(host):\(port)\(path)") else { return (0, nil) }
        var request = URLRequest(url: url)
        if authenticated, let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        do {
            let (data, response) = try await session.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            return (status, (try? JSONSerialization.jsonObject(with: data)) as? [String: Any])
        } catch {
            return (0, nil)
        }
    }

    /// A page fetch that does not parse JSON: status and content type. Used to find out whether the chat page is served.
    public func page(_ path: String) async -> (status: Int, contentType: String?) {
        guard let url = URL(string: "http://\(host):\(port)\(path)") else { return (0, nil) }
        do {
            let (_, response) = try await session.data(for: URLRequest(url: url))
            let http = response as? HTTPURLResponse
            return (http?.statusCode ?? 0, http?.value(forHTTPHeaderField: "Content-Type"))
        } catch { return (0, nil) }
    }

    public struct Probe: Sendable {
        public var httpReady = false
        public var loadedModelIds: [String] = []
        public var instancePid: Int?
        public var instanceModel: String?
        public var statusReadable = false
        /// nil when the status has no `http.draining` field: this Splash cannot drain.
        public var draining: Bool?
        public var authRequired = false
        public var activity: Activity?
    }

    public struct Activity: Equatable, Codable, Sendable {
        public var activeRequests: Int
        public var queued: Int
        public var generating: Int
        public var waiting: Int
        public var idle: Bool { activeRequests == 0 && queued == 0 && generating == 0 && waiting == 0 }
    }

    public func probe() async -> Probe {
        async let ready = get("/ready", authenticated: false)
        async let models = get("/v1/models")
        async let status = get("/status")
        let (r, m, s) = await (ready, models, status)
        var probe = Probe()
        probe.httpReady = r.status == 200 && (r.json?["status"] as? String) == "ready"
        if m.status == 401 || s.status == 401 { probe.authRequired = true }
        if let data = m.json?["data"] as? [[String: Any]] {
            probe.loadedModelIds = data.compactMap { $0["id"] as? String }
        }
        if s.status == 200, let json = s.json {
            probe.statusReadable = true
            if let instance = json["instance"] as? [String: Any] {
                probe.instancePid = instance["pid"] as? Int
                probe.instanceModel = instance["model"] as? String
            }
            probe.activity = Self.activity(from: json)
            probe.draining = (json["http"] as? [String: Any])?["draining"] as? Bool
        }
        return probe
    }

    static func activity(from json: [String: Any]) -> Activity? {
        guard let http = json["http"] as? [String: Any], let requests = http["requests"] as? [String: Any],
              let active = requests["active"] as? Int,
              let scheduler = json["scheduler"] as? [String: Any],
              let admission = json["admission"] as? [String: Any] else { return nil }
        func n(_ d: [String: Any], _ keys: [String]) -> Int { keys.reduce(0) { $0 + ((d[$1] as? Int) ?? 0) } }
        return Activity(
            activeRequests: active,
            queued: n(scheduler, ["queued", "waiting_resources", "waiting_prefix", "waiting_mask"]),
            generating: n(scheduler, ["prefilling", "decoding"]),
            waiting: n(admission, ["waiting", "restoring", "suspended"]))
    }
}
