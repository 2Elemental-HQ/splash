import Foundation
import CryptoKit

/// Where the Splash runtime came from.
public enum SplashSource: String, Codable, Sendable {
    /// The runtime inside this app (`Contents/Resources/Splash`).
    case bundled
    /// A Splash installed on this Mac, for example with Homebrew. Never changed by this app.
    case installed
    /// A path chosen in Settings.
    case custom
}

public struct RuntimeCheck: Equatable, Sendable {
    public enum State: String, Sendable { case verified, failed }
    public var state: State
    public var version: String?
    /// release.json declares drain support. Only a running Splash's own status proves it.
    public var drainDeclared: Bool
    public var problems: [String]
}

/// The runtime that ships inside the app, and the check that it is the one that was shipped.
///
/// The build writes `runtime-manifest.json` after signing: the SHA-256 of every file outside the
/// Python tree and of every Mach-O file inside it. Before a start this recomputes them. A change
/// anywhere else in the bundle is caught by the app's own code signature (`codesign --verify`).
public enum RuntimeBundle {
    public static func root(bundle: Bundle = .main) -> URL? {
        guard let base = bundle.resourceURL?.appendingPathComponent("Splash") else { return nil }
        return FileManager.default.isExecutableFile(atPath: base.appendingPathComponent("bin/splash").path) ? base : nil
    }

    public static func verify(root: URL) -> RuntimeCheck {
        var problems: [String] = []
        func json(_ name: String) -> [String: Any]? {
            (try? Data(contentsOf: root.appendingPathComponent(name)))
                .flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
        }
        guard let manifest = json("runtime-manifest.json"), let files = manifest["files"] as? [String: String] else {
            return RuntimeCheck(state: .failed, version: nil, drainDeclared: false, problems: ["runtime-manifest.json is missing or unreadable"])
        }
        let release = json("release.json")
        let version = release?["version"] as? String
        if let declared = manifest["runtime_version"] as? String, declared != version {
            problems.append("release.json and the manifest name different versions")
        }
        for (relative, expected) in files.sorted(by: { $0.key < $1.key }) {
            guard relative.range(of: "^[A-Za-z0-9_./+@ -]+$", options: .regularExpression) != nil, !relative.contains("..") else {
                problems.append("unexpected path in the manifest: \(relative)"); continue
            }
            guard let actual = sha256(of: root.appendingPathComponent(relative)) else { problems.append("missing or unreadable: \(relative)"); continue }
            if actual != expected { problems.append("changed: \(relative)") }
        }
        // release.json describes the engine as shipped, not as upstream built it.
        for (key, path) in [("binary_sha256", "engine/splash"), ("metallib_sha256", "engine/splash.metallib")] {
            if let recorded = release?[key] as? String, files[path] != recorded { problems.append("release.json \(key) does not match \(path)") }
        }
        let features = release?["features"] as? [String] ?? []
        return RuntimeCheck(state: problems.isEmpty ? .verified : .failed, version: version,
                            drainDeclared: features.contains("drain"), problems: Array(problems.prefix(8)))
    }

    static func sha256(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try? handle.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
