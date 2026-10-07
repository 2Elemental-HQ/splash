import Foundation
import Darwin

public struct ProcessIdentity: Codable, Equatable, Sendable {
    public var pid: Int32
    /// Start time in microseconds since 1970, from the kernel.
    public var startMicros: Int64
}

public enum Spawner {
    /// The kernel's start time for a pid, or nil when no such process exists.
    public static func startMicros(of pid: Int32) -> Int64? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.size
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        guard sysctl(&mib, 4, &info, &size, nil, 0) == 0, size > 0, info.kp_proc.p_pid == pid else { return nil }
        let t = info.kp_proc.p_un.__p_starttime
        return Int64(t.tv_sec) * 1_000_000 + Int64(t.tv_usec)
    }

    public static func identity(of pid: Int32) -> ProcessIdentity? {
        startMicros(of: pid).map { ProcessIdentity(pid: pid, startMicros: $0) }
    }

    public static func isAlive(_ identity: ProcessIdentity) -> Bool {
        startMicros(of: identity.pid) == identity.startMicros
    }

    public static func executablePath(of pid: Int32) -> String? {
        var buffer = [CChar](repeating: 0, count: 4096)
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return nil }
        return String(cString: buffer)
    }

    /// Starts a program in its own session, so it survives this app, with
    /// stdin from /dev/null and stdout and stderr in `output`. Every other
    /// descriptor of this app is closed in the child.
    public static func spawn(executable: URL, arguments: [String], environment: [String: String],
                             output: URL) throws -> ProcessIdentity {
        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_addopen(&actions, 1, output.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        posix_spawn_file_actions_adddup2(&actions, 1, 2)

        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // POSIX_SPAWN_CLOEXEC_DEFAULT is 0x4000 in Darwin's spawn.h.
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSID) | 0x4000)

        let argv = ([executable.path] + arguments).map { strdup($0) } + [nil]
        let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }

        var pid: pid_t = 0
        let result = posix_spawn(&pid, executable.path, &actions, &attributes, argv, envp)
        guard result == 0 else {
            throw AppError("spawn_failed", "Could not start \(executable.path): \(String(cString: strerror(result)))", status: 500)
        }
        return ProcessIdentity(pid: pid, startMicros: startMicros(of: pid) ?? 0)
    }

    /// Non-blocking reap. Returns the wait status once the child has exited.
    public static func reap(_ pid: Int32) -> Int32? {
        var status: Int32 = 0
        let result = waitpid(pid, &status, WNOHANG)
        return result == pid ? status : nil
    }

    /// Signals the process only while the pid still names the same process.
    @discardableResult
    public static func signal(_ identity: ProcessIdentity, _ number: Int32) -> Bool {
        guard isAlive(identity) else { return false }
        return kill(identity.pid, number) == 0
    }

    public static func describe(status: Int32) -> String {
        if status & 0x7f == 0 { return "exit code \((status >> 8) & 0xff)" }
        return "signal \(status & 0x7f)"
    }
}
