import XCTest
@testable import SplashManagerCore

@MainActor
final class APITests: XCTestCase {
    func request(_ method: String, _ path: String, token: String? = "t0ken", body: String? = nil,
                 query: [String: String] = [:]) -> HTTPRequest {
        var headers: [String: String] = [:]
        if let token { headers["authorization"] = "Bearer \(token)" }
        if body != nil { headers["content-type"] = "application/json" }
        return HTTPRequest(method: method, path: path, query: query, headers: headers, body: Data((body ?? "").utf8))
    }

    func json(_ response: HTTPResponse) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: response.body) as? [String: Any]) ?? [:]
    }

    func makeAPI(_ h: Harness) throws -> ManagementAPI {
        try h.secrets.write("t0ken", account: SecretAccount.managementToken)
        return ManagementAPI(supervisor: h.supervisor, secrets: h.secrets)
    }

    func testEveryRouteNeedsTheToken() async throws {
        let h = try Harness(); let api = try makeAPI(h)
        for (method, path) in [("GET", "/api/v1/status"), ("GET", "/api/v1/configs"), ("GET", "/api/v1/logs"),
                               ("POST", "/api/v1/start"), ("POST", "/api/v1/stop"), ("POST", "/api/v1/switch"), ("GET", "/nothing")] {
            let none = await api.handle(request(method, path, token: nil))
            XCTAssertEqual(none.status, 401, "\(method) \(path)")
            let wrong = await api.handle(request(method, path, token: "wrong"))
            XCTAssertEqual(wrong.status, 401)
            XCTAssertEqual(wrong.headers["WWW-Authenticate"], "Bearer")
        }
        await h.cleanup()
    }

    func testStatusAndConfigsShape() async throws {
        let h = try Harness(); let api = try makeAPI(h)
        let status = json(await api.handle(request("GET", "/api/v1/status")))
        XCTAssertEqual(status["api_version"] as? Int, 1)
        XCTAssertEqual(status["state"] as? String, "stopped")
        XCTAssertEqual(status["ownership"] as? String, "none")
        let endpoint = status["endpoint"] as? [String: Any]
        XCTAssertEqual(endpoint?["bind_host"] as? String, "127.0.0.1")
        XCTAssertEqual(endpoint?["requires_api_key"] as? Bool, false)
        let configs = json(await api.handle(request("GET", "/api/v1/configs")))["configs"] as? [[String: Any]]
        XCTAssertEqual(configs?.count, 2)
        XCTAssertEqual(configs?.first?["availability"] as? String, "local")
        XCTAssertEqual(configs?.last?["availability"] as? String, "not_local")
        // The API never reveals a secret.
        let text = String(decoding: try JSONSerialization.data(withJSONObject: status), as: UTF8.self)
        XCTAssertFalse(text.contains("t0ken"))
        await h.cleanup()
    }

    func testBodiesAcceptConfigIdsOnly() async throws {
        let h = try Harness(); let api = try makeAPI(h)
        for body in ["{\"config_id\":\"a\",\"args\":[\"--api-key\",\"x\"]}", "{\"model\":\"x/y\"}", "{\"config_id\":\"a\",\"command\":\"ls\"}",
                     "[1]", "not json", "{\"config_id\":5}", "{\"config_id\":\"a\",\"allow_download\":\"yes\"}",
                     "{\"config_id\":\"a\",\"wait_ready_seconds\":9999}"] {
            let r = await api.handle(request("POST", "/api/v1/start", body: body))
            XCTAssertEqual(r.status, 400, body)
        }
        let stopWithBody = await api.handle(request("POST", "/api/v1/stop", body: "{\"force\":true}"))
        XCTAssertEqual(stopWithBody.status, 400, "stop has no force option")
        let unknown = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"zzz\"}"))
        XCTAssertEqual(unknown.status, 404)
        XCTAssertEqual(h.startLog.count, 0)
        await h.cleanup()
    }

    func testStartWaitsForReadiness() async throws {
        let h = try Harness(control: ["ready_delay": 1.0]); let api = try makeAPI(h)
        h.supervisor.begin()
        let r = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"a\",\"wait_ready_seconds\":30}"))
        XCTAssertEqual(r.status, 200)
        let body = json(r)
        XCTAssertEqual(body["state"] as? String, "ready")
        XCTAssertEqual((body["readiness"] as? [String: Any])?["model_loaded"] as? Bool, true)
        // Without waiting the answer is 202 while the model loads.
        let stopped = await api.handle(request("POST", "/api/v1/stop", body: "{\"wait_stopped_seconds\":20}"))
        XCTAssertEqual(stopped.status, 200)
        try await h.setControl(["ready_delay": 1.0])
        let accepted = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"a\"}"))
        XCTAssertEqual(accepted.status, 202)
        XCTAssertEqual(json(accepted)["state"] as? String, "starting")
        await h.cleanup()
    }

    func testStopOverTheApiDrainsAndAnswersAccepted() async throws {
        let h = try Harness(control: ["hold": 1.5]); let api = try makeAPI(h)
        h.supervisor.begin()
        _ = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"a\",\"wait_ready_seconds\":30}"))
        await h.expect { h.supervisor.drainSupported == true }
        let call = Task { await h.chat() }
        await h.expect { h.supervisor.activity?.activeRequests == 1 }
        let r = await api.handle(request("POST", "/api/v1/stop"))
        XCTAssertEqual(r.status, 202)
        XCTAssertEqual(json(r)["state"] as? String, "stopping")
        let second = await api.handle(request("POST", "/api/v1/switch", body: "{\"config_id\":\"b\",\"allow_download\":true}"))
        XCTAssertEqual(second.status, 409)
        XCTAssertEqual((json(second)["error"] as? [String: Any])?["code"] as? String, "busy")
        let finished = await call.value
        XCTAssertEqual(finished, 200)
        let done = await api.handle(request("GET", "/api/v1/status", query: ["wait_for": "stopped", "timeout": "20"]))
        XCTAssertEqual(json(done)["state"] as? String, "stopped", "\(json(done)) \(h.logs.exportText)")
        await h.cleanup()
    }

    func testStatusSeparatesSavedFromAppliedAndDoesNotCallASnapshotSafe() async throws {
        let h = try Harness(control: ["no_drain": true]); let api = try makeAPI(h)
        h.supervisor.begin()
        _ = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"a\",\"wait_ready_seconds\":30}"))
        await h.expect { h.supervisor.drainSupported == false }
        var config = h.store.config(id: "a")!; config.options.maxContext = "64K"; try h.store.save(config)
        let status = json(await api.handle(request("GET", "/api/v1/status")))
        XCTAssertEqual((status["pending_changes"] as? [String: Any])?["restart_required"] as? Bool, true)
        XCTAssertEqual((status["applied"] as? [String: Any])?["revision"] as? String, commitA)
        let activity = status["activity"] as? [String: Any]
        XCTAssertEqual(activity?["idle"] as? Bool, true)
        XCTAssertEqual(activity?["switch_safe"] as? Bool, false, "idle now, but this Splash cannot refuse new calls")
        for path in ["/api/v1/stop", "/api/v1/switch"] {
            let r = await api.handle(request("POST", path, body: path.hasSuffix("switch") ? "{\"config_id\":\"a\"}" : nil))
            XCTAssertEqual(r.status, 409, path)
            XCTAssertEqual((json(r)["error"] as? [String: Any])?["code"] as? String, "drain_unsupported")
        }
        let start = await api.handle(request("POST", "/api/v1/start", body: "{\"config_id\":\"a\"}"))
        XCTAssertEqual((json(start)["error"] as? [String: Any])?["code"] as? String, "configuration_changed")
        await h.cleanup()
    }

    func testMethodAndRouteErrors() async throws {
        let h = try Harness(); let api = try makeAPI(h)
        let wrongMethod = await api.handle(request("GET", "/api/v1/start"))
        XCTAssertEqual(wrongMethod.status, 405)
        let missing = await api.handle(request("GET", "/api/v1/shell"))
        XCTAssertEqual(missing.status, 404)
        await h.cleanup()
    }

    func testLogsAreRedacted() async throws {
        let h = try Harness(); let api = try makeAPI(h)
        h.logs.append("Authorization: Bearer abcdef0123456789abcdef", source: .splash)
        let r = json(await api.handle(request("GET", "/api/v1/logs", query: ["lines": "5"])))
        let text = String(decoding: try JSONSerialization.data(withJSONObject: r), as: UTF8.self)
        XCTAssertFalse(text.contains("abcdef0123456789"))
        await h.cleanup()
    }
}

final class HTTPServerTests: XCTestCase {
    func testServesAndRejectsGarbage() async throws {
        let port = await MainActor.run { Harness.freePort() }
        let server = HTTPServer { request in HTTPResponse.json(200, ["path": request.path, "len": request.body.count]) }
        server.start(bindings: [.init(host: "127.0.0.1", port: port)])
        defer { server.stop() }
        XCTAssertEqual(server.boundTo.count, 1)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/a/b?x=1")!)
        request.httpMethod = "POST"; request.httpBody = Data("hello".utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual((try JSONSerialization.jsonObject(with: data) as? [String: Any])?["len"] as? Int, 5)

        // Raw garbage gets 400 and the connection closes.
        let reply = try await raw(port: port, send: "NOT HTTP\r\n\r\n")
        XCTAssertTrue(reply.hasPrefix("HTTP/1.1 400"), reply)
        // An oversized declared body gets 413 without reading it.
        let big = try await raw(port: port, send: "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n")
        XCTAssertTrue(big.hasPrefix("HTTP/1.1 413"), big)
    }

    func testRefusesToBindWhenAddressIsNotLocal() {
        let server = HTTPServer { _ in HTTPResponse.json(200, [:]) }
        server.start(bindings: [.init(host: "203.0.113.9", port: 45678)])
        XCTAssertTrue(server.boundTo.isEmpty)
        XCTAssertFalse(server.failures.isEmpty)
    }

    private func raw(port: Int, send text: String) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                let fd = socket(AF_INET, SOCK_STREAM, 0)
                var address = sockaddr_in()
                address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size); address.sin_family = sa_family_t(AF_INET)
                address.sin_port = in_port_t(port).bigEndian; address.sin_addr.s_addr = inet_addr("127.0.0.1")
                let ok = withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
                guard ok == 0 else { continuation.resume(throwing: CocoaError(.fileReadUnknown)); return }
                _ = text.withCString { write(fd, $0, strlen($0)) }
                var buffer = [UInt8](repeating: 0, count: 4096)
                let n = read(fd, &buffer, buffer.count)
                close(fd)
                continuation.resume(returning: String(decoding: buffer.prefix(max(0, n)), as: UTF8.self))
            }
        }
    }
}
