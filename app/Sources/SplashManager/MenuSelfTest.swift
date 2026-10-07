import AppKit
import SplashManagerCore

/// `SplashManager --selftest-menu --selftest-report FILE [--selftest-hold SECONDS]`
///
/// Drives the real NSMenu of the real app, with the real supervisor polling, and writes what it saw. It opens
/// the menu in tracking mode, keeps it open while an outside client changes Splash's state (so the state really
/// changes while the menu is open), then chooses a model in the Model submenu the way a click does. It checks that
/// the menu neither closed nor rebuilt by itself, that the Model submenu stayed the same object, and that the
/// choice reached the saved settings. It is not a physical mouse click: macOS lets no tool send those without the
/// Accessibility permission, which an unattended test cannot be granted.
@MainActor
enum MenuSelfTest {
    static func argument(_ name: String) -> String? {
        guard let i = CommandLine.arguments.firstIndex(of: name), i + 1 < CommandLine.arguments.count else { return nil }
        return CommandLine.arguments[i + 1]
    }

    static func run(menu c: StatusMenuController, env: AppEnvironment) {
        let hold = Double(argument("--selftest-hold") ?? "20") ?? 20
        let reportPath = argument("--selftest-report")
        var problems: [String] = []
        var states: [String] = []
        var report: [String: Any] = [:]
        var submenuAtOpen: ObjectIdentifier?
        var changes = 0
        var chosen = false

        let sampler = Timer(timeInterval: 0.2, repeats: true) { _ in
            MainActor.assumeIsolated {
                let state = env.supervisor.state.rawValue
                if states.last != state { states.append(state); c.note("state -> \(state)") }
            }
        }
        RunLoop.main.add(sampler, forMode: .common)
        let observer = env.supervisor.objectWillChange.sink { _ in changes += 1 }

        func finish() {
            sampler.invalidate()
            observer.cancel()
            report["problems"] = problems
            report["states_seen_while_open"] = states
            report["supervisor_change_announcements"] = changes
            report["timeline"] = c.events
            report["menu_opens"] = c.opens; report["menu_closes"] = c.closes; report["menu_rebuilds"] = c.rebuilds
            report["passed"] = problems.isEmpty
            if let reportPath, let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: URL(fileURLWithPath: reportPath))
            }
            exit(problems.isEmpty ? 0 : 1)
        }

        let openDelay = 2.0
        // The click starts from a run loop timer, not from a main-queue block: a nested tracking loop started inside
        // a block would hold the serial main queue, and nothing else (the supervisor's polling included) could run
        // while the menu is open. A real click arrives as an event and has no such block around it.
        let opener = Timer(timeInterval: openDelay, repeats: false) { _ in
            c.openForTest()   // returns when the menu closes
            // Back from tracking: the menu has closed, by our choice below if all went well.
            if !chosen { problems.append("the menu closed before the choice was made; timeline: " + c.events.suffix(6).joined(separator: " | ")) }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                if c.closes != 1 { problems.append("the menu closed \(c.closes) times, expected exactly once (by the chosen item)") }
                finish()
            }
        }
        RunLoop.main.add(opener, forMode: .common)
        DispatchQueue.main.asyncAfter(deadline: .now() + openDelay + 1) {
            c.note("check +1s: isOpen=\(c.isOpen) state=\(env.supervisor.state.rawValue)")
            if !c.isOpen { problems.append("the menu is not open one second after it was opened") }
            submenuAtOpen = c.menu.item(withTitle: "Model")?.submenu.map(ObjectIdentifier.init)
            if submenuAtOpen == nil { problems.append("no Model submenu") }
            states = [env.supervisor.state.rawValue]
            changes = 0
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + openDelay + 1 + hold) {
            c.note("check end: isOpen=\(c.isOpen) state=\(env.supervisor.state.rawValue)")
            if !c.isOpen { problems.append("the menu closed by itself while open") }
            if c.rebuilds != 1 { problems.append("the menu was rebuilt \(c.rebuilds) times while open") }
            if c.closes != 0 { problems.append("the menu closed \(c.closes) times before a choice was made") }
            let sub = c.menu.item(withTitle: "Model")?.submenu
            if sub.map(ObjectIdentifier.init) != submenuAtOpen { problems.append("the Model submenu was replaced while open") }
            guard let sub, sub.items.count >= 2 else { problems.append("fewer than two models to choose from"); c.menu.cancelTracking(); return }
            // Choose the model that is not selected, as a click on it would.
            let current = env.store.settings.selectedConfigId
            guard let index = sub.items.firstIndex(where: { ($0.representedObject as? String) != current && $0.representedObject != nil }),
                  let wanted = sub.items[index].representedObject as? String else { problems.append("no other model to choose"); c.menu.cancelTracking(); return }
            report["chosen"] = wanted
            chosen = true
            sub.performActionForItem(at: index)
            if env.store.settings.selectedConfigId != wanted { problems.append("the chosen model \(wanted) was not saved as selected") }
            c.menu.cancelTracking()
        }
    }
}
