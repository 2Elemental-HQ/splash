import AppKit
import SplashManagerCore

/// `SplashManager --selftest-menu --selftest-report FILE [--selftest-hold SECONDS]`
///
/// Runs the real app's menu controller against the real supervisor and writes what it saw. The menu is "opened" the way AppKit
/// opens it (`menuNeedsUpdate`, then `menuWillOpen`, on the real NSMenu), kept open for `hold` seconds while an outside client
/// starts Splash through the management API, so the supervisor's state really changes and announces changes under the open menu.
/// Then the other model is chosen in the Model submenu through the item's own action, as a click triggers it, and the menu is
/// "closed" and opened again.
///
/// It checks: no rebuild and no mutation while open; the Model submenu stays the same object; the supervisor's announcements
/// did not reach the menu; the choice is saved; the next opening shows the new state and the choice.
///
/// What it does not do: show the menu on screen or click it. An NSMenu presented by a program with no mouse button down is
/// dismissed by macOS after a second or two (the popup window resigns key by itself, with or without any state change), so
/// a held-open real menu cannot be driven from here without the Accessibility permission. Hovering Model while Splash starts is
/// therefore still a check for a person.
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
        var changes = 0
        var submenuAtOpen: ObjectIdentifier?
        var itemsAtOpen: [ObjectIdentifier] = []
        var titlesAtOpen: [String] = []

        let observer = env.supervisor.objectWillChange.sink { _ in changes += 1 }
        let sampler = Timer(timeInterval: 0.2, repeats: true) { _ in
            MainActor.assumeIsolated {
                let state = env.supervisor.state.rawValue
                if states.last != state { states.append(state); c.note("state -> \(state)") }
            }
        }
        RunLoop.main.add(sampler, forMode: .common)

        func finish() {
            sampler.invalidate()
            observer.cancel()
            report["problems"] = problems
            report["states_seen_while_open"] = states
            report["supervisor_change_announcements_while_open"] = changes
            report["menu_opens"] = c.opens; report["menu_closes"] = c.closes; report["menu_rebuilds"] = c.rebuilds
            report["timeline"] = c.events.map { $0.split(separator: "\n").first.map(String.init) ?? $0 }
            report["passed"] = problems.isEmpty
            if let reportPath, let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: URL(fileURLWithPath: reportPath))
            }
            exit(problems.isEmpty ? 0 : 1)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            c.simulateOpen()
            if !c.isOpen { problems.append("menuWillOpen did not mark the menu open") }
            itemsAtOpen = c.menu.items.map(ObjectIdentifier.init)
            titlesAtOpen = c.menu.items.map(\.title)
            submenuAtOpen = c.menu.item(withTitle: "Model")?.submenu.map(ObjectIdentifier.init)
            if submenuAtOpen == nil { problems.append("no Model submenu") }
            states = [env.supervisor.state.rawValue]
            changes = 0
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 2 + hold) {
            // AppKit may ask again to update a menu that is open (it does for key equivalents); the controller must refuse.
            c.menuNeedsUpdate(c.menu)
            if c.rebuilds != 1 { problems.append("the menu was rebuilt \(c.rebuilds - 1) time(s) while open") }
            if c.menu.items.map(ObjectIdentifier.init) != itemsAtOpen { problems.append("the menu's items were replaced while open") }
            if c.menu.items.map(\.title) != titlesAtOpen { problems.append("a menu title changed while open") }
            let sub = c.menu.item(withTitle: "Model")?.submenu
            if sub.map(ObjectIdentifier.init) != submenuAtOpen { problems.append("the Model submenu was replaced while open") }
            if states.count < 2 { problems.append("the supervisor's state did not change while the menu was open (\(states)); the test proves nothing without a change") }
            if changes == 0 { problems.append("the supervisor announced no change while open") }

            guard let sub, sub.items.count >= 2 else { problems.append("fewer than two models to choose from"); finish(); return }
            let current = env.store.settings.selectedConfigId
            guard let index = sub.items.firstIndex(where: { ($0.representedObject as? String) != current && $0.representedObject != nil }),
                  let wanted = sub.items[index].representedObject as? String else { problems.append("no other model to choose"); finish(); return }
            report["chosen"] = wanted
            // The item's own action, as a click on it triggers it.
            sub.performActionForItem(at: index)
            if env.store.settings.selectedConfigId != wanted { problems.append("the chosen model \(wanted) was not saved as selected") }
            c.simulateClose()
            if c.isOpen { problems.append("the menu is still marked open after it closed") }

            // The next opening shows what happened meanwhile.
            c.simulateOpen()
            let again = c.menu.item(withTitle: "Model")?.submenu
            let checked = again?.items.first(where: { $0.state == .on })?.representedObject as? String
            if checked != wanted { problems.append("the next opening checks \(checked ?? "nothing"), not the chosen model \(wanted)") }
            if c.rebuilds != 2 { problems.append("expected exactly one rebuild for the second opening, saw \(c.rebuilds - 1)") }
            let first = c.menu.items.first?.title ?? ""
            report["first_line_on_second_open"] = first
            c.simulateClose()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { finish() }
        }
    }
}
