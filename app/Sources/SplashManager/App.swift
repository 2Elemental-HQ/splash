import SwiftUI
import AppKit
import ServiceManagement
import SplashManagerCore

@MainActor
final class AppEnvironment: ObservableObject {
    static let shared = AppEnvironment()

    let store: ConfigStore
    let secrets = KeychainStore(service: ProcessInfo.processInfo.environment["SPLASH_MANAGER_KEYCHAIN_SERVICE"] ?? "net.2elemental.splash-manager")
    let logs = LogBuffer()
    let supervisor: SplashSupervisor
    let api: ManagementAPI
    let service: ManagementService

    private init() {
        store = ConfigStore(paths: .default)
        supervisor = SplashSupervisor(store: store, secrets: secrets, logs: logs)
        api = ManagementAPI(supervisor: supervisor, secrets: secrets)
        service = ManagementService(api: api, store: store)
        _ = try? Secrets.ensure(SecretAccount.managementToken, in: secrets)
        if store.configs.isEmpty { importInstalledModels() }
        supervisor.begin()
        service.begin()
    }

    /// Finds the models Splash itself installed and adds the new ones. See `ConfigStore.importSplashInstalledModels`.
    @discardableResult
    func importInstalledModels() -> ImportResult { store.importSplashInstalledModels() }

    func managementToken() -> String { secrets.read(SecretAccount.managementToken) ?? "" }

    func regenerate(_ account: String) {
        try? secrets.write(Secrets.generate(), account: account)
        objectWillChange.send()
    }

    static func setLaunchAtLogin(_ on: Bool) throws {
        if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusMenu: StatusMenuController?
    var windowController: MainWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let env = AppEnvironment.shared
        let windows = MainWindowController(env: env)
        let menu = StatusMenuController(env: env)
        menu.openWindow = { windows.show() }
        windowController = windows
        statusMenu = menu
        if CommandLine.arguments.contains("--selftest-menu") { MenuSelfTest.run(menu: menu, env: env) }
    }

    /// A second launch (or a click on the app in Finder) opens the window.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        windowController?.show()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let env = AppEnvironment.shared
        guard env.supervisor.ownership == .managed, env.store.settings.stopSplashWhenAppQuits else {
            env.service.stop()
            return .terminateNow
        }
        Task { @MainActor in
            let check = await env.supervisor.checkIdle()
            let drains = env.supervisor.drainSupported == true
            var mode = drains ? "drain" : "force"
            if !check.safe || !drains {
                let alert = NSAlert()
                alert.messageText = check.safe ? "Splash cannot drain" : "Splash is busy"
                alert.informativeText = (check.safe ? "This Splash build cannot finish running calls before it stops. " : (check.reason ?? "Calls may be running.") + " ")
                    + "Quitting can end calls."
                alert.addButton(withTitle: "Cancel")
                if drains { alert.addButton(withTitle: "Let calls finish, then quit") }
                alert.addButton(withTitle: "Stop now and quit")
                NSApp.activate(ignoringOtherApps: true)
                let answer = alert.runModal()
                if answer == .alertFirstButtonReturn { sender.reply(toApplicationShouldTerminate: false); return }
                mode = (drains && answer == .alertSecondButtonReturn) ? "drain" : "force"
            }
            if mode == "drain" {
                _ = try? await env.supervisor.stop(force: false)
                _ = await env.supervisor.wait(for: .stopped, timeout: 3600)
            } else {
                _ = try? await env.supervisor.stop(force: true)
            }
            env.service.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

@MainActor
final class MainWindowController {
    private var window: NSWindow?
    private let env: AppEnvironment

    init(env: AppEnvironment) { self.env = env }

    func show() {
        if window == nil {
            let root = MainWindow()
                .environmentObject(env).environmentObject(env.supervisor).environmentObject(env.store)
                .environmentObject(env.logs).environmentObject(env.service)
                .frame(minWidth: 780, minHeight: 520)
            let created = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 920, height: 640),
                                   styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
            created.title = "Splash Manager"
            created.isReleasedWhenClosed = false
            created.contentView = NSHostingView(rootView: root)
            created.center()
            created.setFrameAutosaveName("SplashManagerMain")
            window = created
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }
}

@main
struct SplashManagerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    init() {
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
        if let index = CommandLine.arguments.firstIndex(of: "--snapshot"), index + 1 < CommandLine.arguments.count {
            Snapshot.run(into: CommandLine.arguments[index + 1])
        }
    }

    // The menu bar item and the window are AppKit (see StatusMenu.swift, MainWindowController); the app needs one scene.
    var body: some Scene {
        Settings { EmptyView() }
    }
}
