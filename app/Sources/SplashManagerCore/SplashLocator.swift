import Foundation

/// What the installed Splash really offers. Read from the binary, never assumed.
public struct SplashInstall: Equatable, Sendable {
    public var executable: URL
    public var version: String
    /// Long options in `splash serve --help`.
    public var serveFlags: Set<String>

    public func supports(_ flag: String) -> Bool { serveFlags.contains(flag) }
}

public enum SplashLocator {
    public static let candidates = ["/opt/homebrew/bin/splash", "/usr/local/bin/splash"]

    public static func find(preferred: String?) -> URL? {
        let fm = FileManager.default
        var paths = [String]()
        if let preferred, !preferred.isEmpty { paths.append(preferred) }
        paths.append(contentsOf: candidates)
        for path in paths where fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    /// Runs `splash --version` and `splash serve --help` with a small timeout.
    public static func inspect(_ executable: URL) async -> SplashInstall? {
        async let versionOut = run(executable, ["--version"])
        async let helpOut = run(executable, ["serve", "--help"])
        guard let versionText = await versionOut, let help = await helpOut else { return nil }
        let version = versionText.split(whereSeparator: \.isNewline).first.map(String.init)?
            .replacingOccurrences(of: "Splash ", with: "") ?? "unknown"
        return SplashInstall(executable: executable, version: version, serveFlags: parseFlags(help))
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
