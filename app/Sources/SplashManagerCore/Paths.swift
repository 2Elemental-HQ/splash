import Foundation

public struct AppPaths: Sendable {
    public let support: URL
    public let runDirectory: URL
    public let logsDirectory: URL

    public static let `default`: AppPaths = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return AppPaths(
            support: home.appendingPathComponent("Library/Application Support/Splash Manager", isDirectory: true),
            logs: home.appendingPathComponent("Library/Logs/Splash Manager", isDirectory: true))
    }()

    public init(support: URL, logs: URL) {
        self.support = support
        self.runDirectory = support.appendingPathComponent("run", isDirectory: true)
        self.logsDirectory = logs
    }

    public var configFile: URL { support.appendingPathComponent("config.json") }
    public var managedFile: URL { runDirectory.appendingPathComponent("managed.json") }
    /// The running Splash's console output. Truncated at every start and
    /// removed after a clean stop, so a crash leaves only the last run.
    public var consoleFile: URL { runDirectory.appendingPathComponent("console.out") }
    public var persistentLog: URL { logsDirectory.appendingPathComponent("splash-manager.log") }

    public func prepare() throws {
        let fm = FileManager.default
        for dir in [support, runDirectory] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true,
                                   attributes: [.posixPermissions: 0o700])
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: dir.path)
        }
    }
}

enum Files {
    static func writeAtomically(_ data: Data, to url: URL, mode: Int = 0o600) throws {
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString.prefix(8)).tmp")
        guard FileManager.default.createFile(atPath: temporary.path, contents: data,
                                             attributes: [.posixPermissions: mode]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }
}
