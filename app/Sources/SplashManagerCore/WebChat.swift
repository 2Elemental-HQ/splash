import Foundation

/// What "Open Web Chat" should do in the current state. Pure, so the menu and the window agree and tests can check it.
public enum WebChatAction: Equatable, Sendable {
    /// The chat page is up: open this local address (no key in it; the page asks for one when needed).
    case open(URL)
    /// Nothing runs: start the selected model, then open the page.
    case startThenOpen
    /// Splash is starting or stopping: not yet.
    case wait(String)
    /// It cannot be opened, and why.
    case unavailable(String)

    public var isEnabled: Bool {
        switch self {
        case .open, .startThenOpen: true
        case .wait, .unavailable: false
        }
    }

    public var menuTitle: String {
        switch self {
        case .open: "Open Web Chat…"
        case .startThenOpen: "Start Splash and Open Web Chat…"
        case .wait(let why): "Open Web Chat (\(why))"
        case .unavailable: "Open Web Chat (not available)"
        }
    }

    public static func decide(_ status: ManagerStatus, hasSelectedModel: Bool) -> WebChatAction {
        switch (status.ownership, status.state) {
        case (_, .starting): return .wait("Splash is starting…")
        case (_, .stopping): return .wait("Splash is stopping…")
        case (_, .ready):
            guard let text = status.webChat.url, let url = URL(string: text) else { return .wait("checking…") }
            switch status.webChat.available {
            case true?: return .open(url)
            case false?: return .unavailable("This Splash does not serve its chat page. It was started without it, or it is a build that has none.")
            case nil: return .wait("checking…")
            }
        case (_, .failed): return .unavailable("Splash failed to start. See Overview.")
        case (_, .stopped):
            return hasSelectedModel ? .startThenOpen : .unavailable("Add a model first.")
        }
    }
}
