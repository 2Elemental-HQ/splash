import Foundation

/// What the installed Splash really offers. Read from the binary, never assumed.
public struct SplashInstall: Equatable, Sendable {
    public var executable: URL
    public var version: String
    public var source: SplashSource = .installed
    /// Set for the bundled runtime: the result of checking it against its manifest.
    public var runtime: RuntimeCheck?
    /// Long options in `splash serve --help`.
    public var serveFlags: Set<String>
    /// The model families and draft repositories the installed Splash declares
    /// (`install/families.py`), or nil when they could not be read.
    public var families: [SplashFamily]?

    public func supports(_ flag: String) -> Bool { serveFlags.contains(flag) }
}

public enum SplashLocator {
    public static let candidates = ["/opt/homebrew/bin/splash", "/usr/local/bin/splash"]

    /// Chooses the Splash to run: a path set in Settings first; then the runtime bundled with the app
    /// (it can drain, so it is the default); then a Splash installed on this Mac. `preferInstalled`
    /// puts the installed one before the bundled one. Nothing here changes any installation.
    public static func resolve(preferred: String?, preferInstalled: Bool = false, bundledRoot: URL? = RuntimeBundle.root())
        -> (url: URL, source: SplashSource)? {
        let fm = FileManager.default
        if let preferred, !preferred.isEmpty, fm.isExecutableFile(atPath: preferred) {
            return (URL(fileURLWithPath: preferred), .custom)
        }
        let bundled = bundledRoot.map { ($0.appendingPathComponent("bin/splash"), SplashSource.bundled) }
        let installed = candidates.first(where: { fm.isExecutableFile(atPath: $0) }).map { (URL(fileURLWithPath: $0), SplashSource.installed) }
        let order = preferInstalled ? [installed, bundled] : [bundled, installed]
        return order.compactMap { $0 }.first.map { (url: $0.0, source: $0.1) }
    }

    public static func find(preferred: String?) -> URL? { resolve(preferred: preferred, bundledRoot: nil)?.url }

    /// Runs `splash --version` and `splash serve --help` with a small timeout.
    public static func inspect(_ executable: URL, source: SplashSource = .installed) async -> SplashInstall? {
        async let versionOut = run(executable, ["--version"])
        async let helpOut = run(executable, ["serve", "--help"])
        guard let versionText = await versionOut, let help = await helpOut else { return nil }
        let version = versionText.split(whereSeparator: \.isNewline).first.map(String.init)?
            .replacingOccurrences(of: "Splash ", with: "") ?? "unknown"
        let families = await SplashFamilies.load(executable: executable)
        return SplashInstall(executable: executable, version: version, source: source, serveFlags: parseFlags(help), families: families)
    }

    static func parseFlags(_ help: String) -> Set<String> {
        guard let regex = try? NSRegularExpression(pattern: "(?m)^\\s+(--[a-z][a-z0-9-]*)") else { return [] }
        let range = NSRange(help.startIndex..., in: help)
        return Set(regex.matches(in: help, range: range).compactMap {
            Range($0.range(at: 1), in: help).map { String(help[$0]) }
        })
    }

    static func run(_ executable: URL, _ arguments: [String], timeout: TimeInterval = 20) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                let process = Process()
                process.executableURL = executable
                process.arguments = arguments
                var environment = ProcessInfo.processInfo.environment
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                process.environment = environment
                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = FileHandle.nullDevice
                process.standardInput = FileHandle.nullDevice
                do { try process.run() } catch { continuation.resume(returning: nil); return }
                let killer = DispatchWorkItem { if process.isRunning { process.terminate() } }
                DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: killer)
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                killer.cancel()
                continuation.resume(returning: process.terminationStatus == 0 ? String(decoding: data, as: UTF8.self) : nil)
            }
        }
    }
}

public struct SplashFamily: Equatable, Sendable {
    public var name: String
    public var draftRepo: String
    public init(name: String, draftRepo: String) { self.name = name; self.draftRepo = draftRepo }
}

/// Reads the family table of the installed Splash instead of copying it: the
/// app follows `install/families.py` of whatever release is installed.
public enum SplashFamilies {
    /// The directory that holds `install/families.py`: the Homebrew `libexec`, or a source checkout.
    static func root(of executable: URL) -> URL? {
        let fm = FileManager.default
        let real = executable.resolvingSymlinksInPath()
        let bin = real.deletingLastPathComponent()
        let candidates = [bin.deletingLastPathComponent().appendingPathComponent("libexec"), bin, bin.deletingLastPathComponent()]
        return candidates.first { fm.fileExists(atPath: $0.appendingPathComponent("install/families.py").path) }
    }

    public static func load(executable: URL) async -> [SplashFamily]? {
        guard let root = root(of: executable) else { return nil }
        let fm = FileManager.default
        let pythons = [root.appendingPathComponent("python/bin/python3"), root.appendingPathComponent(".venv/bin/python"),
                       URL(fileURLWithPath: "/usr/bin/python3")]
        guard let python = pythons.first(where: { fm.isExecutableFile(atPath: $0.path) }) else { return nil }
        let code = "import json; from install import families as f; print(json.dumps([[x.name, x.draft_repo] for x in f.FAMILIES]))"
        guard let text = await run(python, root: root, ["-c", code]), let data = text.data(using: .utf8),
              let rows = (try? JSONSerialization.jsonObject(with: data)) as? [[String]] else { return nil }
        let families = rows.compactMap { $0.count == 2 ? SplashFamily(name: $0[0], draftRepo: $0[1]) : nil }
        return families.isEmpty ? nil : families
    }

    private static func run(_ python: URL, root: URL, _ arguments: [String]) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                let process = Process()
                process.executableURL = python
                process.arguments = arguments
                process.currentDirectoryURL = root
                process.environment = ["PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": root.path]
                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = FileHandle.nullDevice
                process.standardInput = FileHandle.nullDevice
                do { try process.run() } catch { continuation.resume(returning: nil); return }
                let killer = DispatchWorkItem { if process.isRunning { process.terminate() } }
                DispatchQueue.global().asyncAfter(deadline: .now() + 10, execute: killer)
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                killer.cancel()
                continuation.resume(returning: process.terminationStatus == 0 ? String(decoding: data, as: UTF8.self) : nil)
            }
        }
    }
}

extension SplashFamilies {
    /// Models the installed Splash suggests (`install/completions/suggested-models.txt`).
    /// Nothing is downloaded by reading them.
    public static func suggestedModels(executable: URL) -> [String] {
        guard let root = root(of: executable),
              let text = try? String(contentsOf: root.appendingPathComponent("install/completions/suggested-models.txt"), encoding: .utf8)
        else { return [] }
        return text.split(whereSeparator: \.isNewline).compactMap { try? Validation.modelId(String($0)) }
    }
}
