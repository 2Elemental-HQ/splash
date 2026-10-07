import Foundation

/// What this Mac must offer. The numbers come from the README of Splash 1.3.0
/// (Apple M3 or newer, macOS 26.4 or later). Splash's own device check, run at every start,
/// is authoritative; these checks only tell a person early.
public enum SystemRequirements {
    public static let splashMinimumMacOS = OperatingSystemVersion(majorVersion: 26, minorVersion: 4, patchVersion: 0)
    public static let splashMinimumChip = 3
    public static let appMinimumMacOS = OperatingSystemVersion(majorVersion: 14, minorVersion: 0, patchVersion: 0)

    public struct Check: Equatable, Sendable, Identifiable {
        public var id: String
        public var title: String
        /// nil: could not be determined.
        public var ok: Bool?
        public var detail: String
    }

    /// "Apple M4 Max" -> 4. nil for anything else, Intel included.
    public static func chipGeneration(_ brand: String?) -> Int? {
        guard let brand, let range = brand.range(of: "^Apple M([0-9]+)", options: .regularExpression) else { return nil }
        return Int(brand[range].dropFirst("Apple M".count))
    }

    static func atLeast(_ v: OperatingSystemVersion, _ minimum: OperatingSystemVersion) -> Bool {
        (v.majorVersion, v.minorVersion, v.patchVersion) >= (minimum.majorVersion, minimum.minorVersion, minimum.patchVersion)
    }

    public static func evaluate(appleSilicon: Bool, chipBrand: String?, os: OperatingSystemVersion) -> [Check] {
        let osText = "\(os.majorVersion).\(os.minorVersion)"
        var checks = [Check(id: "arch", title: "Apple silicon", ok: appleSilicon,
                            detail: appleSilicon ? "This Mac has Apple silicon." : "Splash and this app need Apple silicon; Intel Macs are not supported.")]
        let generation = chipGeneration(chipBrand)
        checks.append(Check(
            id: "chip", title: "Apple M3 or newer (for Splash)",
            ok: appleSilicon ? generation.map { $0 >= splashMinimumChip } : false,
            detail: generation != nil ? "This Mac has \(chipBrand ?? "an Apple chip")." : "The chip generation could not be read."))
        checks.append(Check(
            id: "os", title: "macOS 26.4 or later (for Splash)", ok: atLeast(os, splashMinimumMacOS),
            detail: "This Mac runs macOS \(osText). The app itself needs macOS 14 or later."))
        return checks
    }

    public static func current() -> [Check] {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        let brand = size > 0 ? String(cString: buffer) : nil
        var arm: Int32 = 0
        var armSize = MemoryLayout<Int32>.size
        sysctlbyname("hw.optional.arm64", &arm, &armSize, nil, 0)
        return evaluate(appleSilicon: arm == 1, chipBrand: brand, os: ProcessInfo.processInfo.operatingSystemVersion)
    }
}

/// Guided installation of Splash through Homebrew. The app never installs Homebrew itself
/// (that script needs administrator rights and the Xcode command line tools).
public enum Homebrew {
    public static func find() -> URL? {
        ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"].map(URL.init(fileURLWithPath:))
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    public static let tap = "incoai/tap/splash"

    /// Runs `brew install incoai/tap/splash` and reports each output line. Returns the exit status.
    public static func installSplash(brew: URL, onLine: @escaping @Sendable (String) -> Void) async -> Int32 {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                let process = Process()
                process.executableURL = brew
                process.arguments = ["install", tap]
                var environment = ProcessInfo.processInfo.environment
                environment["HOMEBREW_NO_ENV_HINTS"] = "1"
                environment["NONINTERACTIVE"] = "1"
                environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
                process.environment = environment
                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe
                process.standardInput = FileHandle.nullDevice
                do { try process.run() } catch { onLine("Could not start brew: \(error)"); continuation.resume(returning: -1); return }
                var pending = Data()
                while true {
                    let chunk = pipe.fileHandleForReading.availableData
                    if chunk.isEmpty { break }
                    pending.append(chunk)
                    while let newline = pending.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                        let line = String(decoding: pending[pending.startIndex..<newline], as: UTF8.self)
                        pending.removeSubrange(pending.startIndex...newline)
                        if !line.isEmpty { onLine(line) }
                    }
                }
                process.waitUntilExit()
                continuation.resume(returning: process.terminationStatus)
            }
        }
    }
}
