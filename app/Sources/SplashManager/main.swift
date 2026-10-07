import AppKit
import SplashManagerCore

// The app is plain AppKit: a status item with an NSMenu, and one window made by hand. A SwiftUI `App` needs a scene, and
// even an empty `Settings` scene creates a window that takes key focus when the app activates; that window pushed the
// open menu away (the menu's own window resigned key the moment it appeared).

// `SplashManager --print-token` prints the management token for scripts and exits.
if CommandLine.arguments.contains("--print-token") {
    let store = KeychainStore(service: ProcessInfo.processInfo.environment["SPLASH_MANAGER_KEYCHAIN_SERVICE"] ?? "net.2elemental.splash-manager")
    print((try? Secrets.ensure(SecretAccount.managementToken, in: store)) ?? "")
    exit(0)
}

// `SplashManager --verify-runtime` checks the bundled Splash runtime against its manifest and exits 0 or 1.
if CommandLine.arguments.contains("--verify-runtime") {
    guard let root = RuntimeBundle.root() else { print("no bundled runtime"); exit(1) }
    let check = RuntimeBundle.verify(root: root)
    print("runtime \(check.version ?? "?"): \(check.state.rawValue)\(check.problems.isEmpty ? "" : " - " + check.problems.joined(separator: "; "))")
    exit(check.state == .verified ? 0 : 1)
}

let delegate = AppDelegate()
let application = NSApplication.shared
application.delegate = delegate
application.setActivationPolicy(.accessory)
application.run()
