import SwiftUI
import AppKit
import SplashManagerCore

/// `SplashManager --snapshot DIR` draws each tab in a real window into a PNG and exits. It exists
/// to check the real views against real state, where the screen cannot be captured.
@MainActor
enum Snapshot {
    static func run(into directory: String) {
        let env = AppEnvironment.shared
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            let tabs: [(String, AnyView)] = [
                ("overview", AnyView(OverviewView())), ("models", AnyView(ModelsView())),
                ("settings", AnyView(SettingsView())), ("logs", AnyView(LogsView())),
            ]
            for (name, view) in tabs {
                let root = NavigationStack { view }
                    .environmentObject(env).environmentObject(env.supervisor).environmentObject(env.store)
                    .environmentObject(env.logs).environmentObject(env.service)
                let host = NSHostingView(rootView: root)
                let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 820, height: 640), styleMask: [.titled],
                                      backing: .buffered, defer: false)
                window.appearance = NSAppearance(named: .aqua)
                window.contentView = host
                window.orderFrontRegardless()
                try? await Task.sleep(nanoseconds: 800_000_000)
                host.layoutSubtreeIfNeeded()
                if let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
                    host.cacheDisplay(in: host.bounds, to: rep)
                    try? rep.representation(using: .png, properties: [:])?
                        .write(to: URL(fileURLWithPath: directory).appendingPathComponent("\(name).png"))
                }
                window.orderOut(nil)
            }
            exit(0)
        }
    }
}
