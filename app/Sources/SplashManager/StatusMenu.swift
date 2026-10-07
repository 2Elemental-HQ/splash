import AppKit
import Combine
import SplashManagerCore

/// The menu bar item, built with AppKit so that its menu is stable.
///
/// A SwiftUI `MenuBarExtra` rebuilds its menu whenever an observed object announces a change, and the supervisor
/// announces many while it polls; that closed the open "Model" submenu. Here the menu is built only in
/// `menuNeedsUpdate` (just before it opens) and never while it is open. Polling changes the icon, not the menu.
@MainActor
final class StatusMenuController: NSObject, NSMenuDelegate {
    private let env: AppEnvironment
    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    let menu = NSMenu()
    private var cancellable: AnyCancellable?
    private var notes: [NSObjectProtocol] = []
    private var bag = Set<AnyCancellable>()
    private(set) var isOpen = false
    /// Counts for the self test: a menu that closes by itself is the bug.
    private(set) var opens = 0
    private(set) var closes = 0
    private(set) var rebuilds = 0
    /// A timeline for the self test: what happened to the menu and when.
    private(set) var events: [String] = []
    private let born = Date()
    func note(_ text: String) { events.append(String(format: "%.2f %@", Date().timeIntervalSince(born), text)) }
    var openWindow: () -> Void = {}

    init(env: AppEnvironment) {
        self.env = env
        super.init()
        menu.delegate = self
        menu.autoenablesItems = false
        item.menu = menu
        item.button?.toolTip = "Splash Manager"
        updateIcon()
        cancellable = env.supervisor.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.updateIcon() }
        // For the self test's timeline: what else moves around the menu.
        for name in [NSApplication.didResignActiveNotification, NSApplication.didBecomeActiveNotification,
                     NSApplication.didUpdateNotification, NSMenu.didEndTrackingNotification, NSMenu.didBeginTrackingNotification,
                     NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification] where name != NSApplication.didUpdateNotification {
            notes.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] n in
                MainActor.assumeIsolated {
                    let who = (n.object as? NSWindow).map { "\(type(of: $0)) title=\($0.title) visible=\($0.isVisible) level=\($0.level.rawValue)" } ?? "\(String(describing: n.object))"
                    self?.note("notification \(n.name.rawValue) \(who)")
                }
            })
        }
        env.supervisor.$state.dropFirst().sink { [weak self] value in self?.note("supervisor.state = \(value.rawValue)") }.store(in: &bag)
    }

    /// What AppKit does around a displayed menu, called by hand for the self test: `menuNeedsUpdate`, then `menuWillOpen`.
    func simulateOpen() { menuNeedsUpdate(menu); menuWillOpen(menu) }
    func simulateClose() { menuDidClose(menu) }

    private func updateIcon() {
        // Changing the item's image while its menu is open can dismiss the menu; do it when it closes.
        guard !isOpen else { return }
        let symbol = env.supervisor.state.symbol
        if item.button?.image?.accessibilityDescription != symbol {
            item.button?.image = NSImage(systemSymbolName: symbol, accessibilityDescription: symbol)
        }
    }

    // MARK: NSMenuDelegate

    func menuNeedsUpdate(_ menu: NSMenu) {
        note("menuNeedsUpdate (open: \(isOpen))")
        guard !isOpen else { return }   // never rebuild what is open
        rebuild(menu)
    }

    func menuWillOpen(_ menu: NSMenu) { isOpen = true; opens += 1; note("menuWillOpen (app active: \(NSApp.isActive))") }
    func menuDidClose(_ menu: NSMenu) { isOpen = false; closes += 1; note("menuDidClose"); updateIcon() }

    // MARK: Building

    func rebuild(_ menu: NSMenu) {
        rebuilds += 1
        menu.removeAllItems()
        let supervisor = env.supervisor
        let status = supervisor.snapshot()
        let selectedId = env.store.settings.selectedConfigId

        func label(_ title: String) -> NSMenuItem {
            let entry = NSMenuItem(title: title, action: nil, keyEquivalent: "")
            entry.isEnabled = false
            return entry
        }
        menu.addItem(label("Splash: \(status.state.title)\(status.ownership == .external ? " (external)" : "")"))
        if let model = supervisor.loadedModelId ?? status.config?.displayName { menu.addItem(label(model)) }
        if status.state == .ready { menu.addItem(label("Endpoint \(status.endpoint.localUrl)/v1")) }
        if let error = supervisor.lastError, supervisor.state == .failed { menu.addItem(label("Error: \(String(error.message.prefix(90)))")) }
        menu.addItem(.separator())

        switch (supervisor.ownership, supervisor.state) {
        case (.managed, .ready), (.managed, .starting):
            menu.addItem(action("Stop Splash", #selector(stopSplash)))
        case (.external, _):
            menu.addItem(label("Started outside this app"))
        case (_, .stopping):
            menu.addItem(label("Stopping…"))
        default:
            let start = action("Start Splash", #selector(startSplash))
            start.isEnabled = selectedId != nil
            menu.addItem(start)
        }

        let modelMenu = NSMenu(title: "Model")
        modelMenu.autoenablesItems = false
        for config in env.store.configs {
            let entry = NSMenuItem(title: config.displayName, action: #selector(selectModel(_:)), keyEquivalent: "")
            entry.target = self
            entry.representedObject = config.id
            entry.state = selectedId == config.id ? .on : .off
            modelMenu.addItem(entry)
        }
        if env.store.configs.isEmpty { modelMenu.addItem(label("No models configured")) }
        let models = NSMenuItem(title: "Model", action: nil, keyEquivalent: "")
        models.submenu = modelMenu
        menu.addItem(models)

        menu.addItem(.separator())
        let chat = WebChatAction.decide(status, hasSelectedModel: selectedId != nil)
        let chatItem = action(chat.menuTitle, #selector(openWebChat))
        chatItem.isEnabled = chat.isEnabled
        if case .unavailable(let why) = chat { chatItem.toolTip = why }
        menu.addItem(chatItem)
        menu.addItem(action("Open Splash Manager…", #selector(showWindow)))
        menu.addItem(.separator())
        let quit = action("Quit", #selector(quit), key: "q")
        menu.addItem(quit)
    }

    private func action(_ title: String, _ selector: Selector, key: String = "") -> NSMenuItem {
        let entry = NSMenuItem(title: title, action: selector, keyEquivalent: key)
        entry.target = self
        return entry
    }

    // MARK: Actions

    @objc func startSplash() { Task { await StartFlow.run(env.supervisor, configId: env.store.settings.selectedConfigId) } }
    @objc func stopSplash() { Task { await StopFlow.run(env.supervisor) } }
    @objc func showWindow() { openWindow() }
    @objc func quit() { NSApp.terminate(nil) }

    @objc func selectModel(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String, let config = env.store.config(id: id) else { return }
        try? env.store.updateSettings { $0.selectedConfigId = config.id }
        // With a model loaded a pick means "switch to it": the switch drains, so running calls finish first.
        if env.supervisor.ownership == .managed, env.supervisor.state == .ready || env.supervisor.state == .starting {
            Task { await SwitchFlow.run(env.supervisor, configId: config.id) }
        }
    }

    @objc func openWebChat() { WebChatLauncher.open(env) }
}

@MainActor
enum WebChatLauncher {
    /// Opens the chat page of the running Splash in the default browser, or starts it first. The address has no key in it.
    static func open(_ env: AppEnvironment) {
        let status = env.supervisor.snapshot()
        switch WebChatAction.decide(status, hasSelectedModel: env.store.settings.selectedConfigId != nil) {
        case .open(let url): NSWorkspace.shared.open(url)
        case .startThenOpen:
            Task {
                await StartFlow.run(env.supervisor, configId: env.store.settings.selectedConfigId)
                let ready = await env.supervisor.wait(for: .ready, timeout: 600)
                if ready.state == .ready {
                    // The page check follows readiness by a moment.
                    for _ in 0..<50 where env.supervisor.webChatAvailable == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
                    if case .open(let url) = WebChatAction.decide(env.supervisor.snapshot(), hasSelectedModel: true) { NSWorkspace.shared.open(url) }
                }
            }
        case .wait(let why): Dialogs.show("Web Chat", "Not yet: \(why)")
        case .unavailable(let why): Dialogs.show("Web Chat is not available", why)
        }
    }
}
