import SwiftUI
import AppKit
import SplashManagerCore

extension RunState {
    var title: String {
        switch self {
        case .stopped: "Stopped"
        case .starting: "Starting"
        case .ready: "Ready"
        case .stopping: "Stopping"
        case .failed: "Failed"
        }
    }
    var color: Color {
        switch self {
        case .stopped: .secondary
        case .starting, .stopping: .orange
        case .ready: .green
        case .failed: .red
        }
    }
    var symbol: String {
        switch self {
        case .stopped: "bolt.slash"
        case .starting, .stopping: "bolt.badge.clock"
        case .ready: "bolt.fill"
        case .failed: "exclamationmark.triangle"
        }
    }
}

struct StatusBadge: View {
    let state: RunState
    var external = false
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(state.color).frame(width: 10, height: 10)
            Text(state.title + (external ? " (external)" : "")).font(.headline)
        }
        .padding(.horizontal, 10).padding(.vertical, 4)
        .background(state.color.opacity(0.15), in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

struct CopyField: View {
    let label: String
    let value: String
    var secret = false
    @State private var revealed = false
    @State private var copied = false

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
            Text(secret && !revealed ? String(repeating: "•", count: 24) : value)
                .font(.system(.body, design: .monospaced)).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
            Spacer()
            if secret { Button(revealed ? "Hide" : "Show") { revealed.toggle() }.buttonStyle(.link) }
            Button(copied ? "Copied" : "Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(value, forType: .string)
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
            }.buttonStyle(.link)
        }
    }
}

func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

extension Availability {
    var title: String {
        switch self {
        case .loaded: "Loaded"
        case .local: "Files in cache"
        case .notLocal: "Needs download"
        }
    }
    var color: Color {
        switch self {
        case .loaded: .green
        case .local: .blue
        case .notLocal: .orange
        }
    }
}

struct Pill: View {
    let text: String
    let color: Color
    var body: some View {
        Text(text).font(.caption.weight(.medium)).padding(.horizontal, 8).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule()).foregroundStyle(color)
    }
}
