import Foundation

public enum Redactor {
    private static let rules: [(NSRegularExpression, String)] = {
        let patterns: [(String, String)] = [
            ("(?i)(authorization:\\s*bearer\\s+)\\S+", "$1[redacted]"),
            ("(?i)bearer\\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [redacted]"),
            ("(?i)(x-api-key:\\s*)\\S+", "$1[redacted]"),
            ("(?i)(--api-key[=\\s]+)\\S+", "$1[redacted]"),
            ("(?i)((?:api[_-]?key|token|secret|password)[\"']?\\s*[:=]\\s*[\"']?)[^\\s\"',}]{4,}", "$1[redacted]"),
            ("\\bhf_[A-Za-z0-9]{10,}\\b", "hf_[redacted]"),
            ("\\bsk-[A-Za-z0-9_-]{10,}\\b", "sk-[redacted]"),
        ]
        return patterns.compactMap { p in (try? NSRegularExpression(pattern: p.0)).map { ($0, p.1) } }
    }()

    public static func redact(_ line: String) -> String {
        var text = line
        for (regex, template) in rules {
            text = regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text),
                                                  withTemplate: template)
        }
        return text
    }
}

public struct LogLine: Identifiable, Equatable, Sendable {
    public enum Source: String, Sendable { case app, splash }
    public let id: Int
    public let date: Date
    public let source: Source
    public let text: String
}

/// The last lines of app and Splash output, in memory, redacted on entry.
@MainActor
public final class LogBuffer: ObservableObject {
    @Published public private(set) var lines: [LogLine] = []
    private var nextId = 0
    private let capacity: Int
    public var persistTo: URL?

    public init(capacity: Int = 2000) { self.capacity = capacity }

    public func append(_ text: String, source: LogLine.Source) {
        let clean = Redactor.redact(String(text.prefix(2000)))
        lines.append(LogLine(id: nextId, date: Date(), source: source, text: clean))
        nextId += 1
        if lines.count > capacity { lines.removeFirst(lines.count - capacity) }
        if let url = persistTo { Self.persist(clean, source: source, to: url) }
    }

    public func clear() { lines.removeAll() }

    public var exportText: String {
        let format = ISO8601DateFormatter()
        return lines.map { "\(format.string(from: $0.date)) [\($0.source.rawValue)] \($0.text)" }.joined(separator: "\n")
    }

    private nonisolated static func persist(_ text: String, source: LogLine.Source, to url: URL) {
        let fm = FileManager.default
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                attributes: [.posixPermissions: 0o700])
        if let size = (try? fm.attributesOfItem(atPath: url.path)[.size] as? Int), size > 2_000_000 {
            let old = url.appendingPathExtension("1")
            try? fm.removeItem(at: old)
            try? fm.moveItem(at: url, to: old)
        }
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        let stamp = ISO8601DateFormatter().string(from: Date())
        try? handle.write(contentsOf: Data("\(stamp) [\(source.rawValue)] \(text)\n".utf8))
    }
}

/// Follows a growing file and reports complete lines.
final class FileTail: @unchecked Sendable {
    private let url: URL
    private var offset: UInt64 = 0
    private var partial = Data()

    init(url: URL) { self.url = url }

    func reset() { offset = 0; partial = Data() }

    func readNewLines() -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        if size < offset { offset = 0; partial = Data() }
        guard size > offset else { return [] }
        try? handle.seek(toOffset: offset)
        let data = (try? handle.read(upToCount: Int(min(size - offset, 1_000_000)))) ?? Data()
        offset += UInt64(data.count)
        partial.append(data)
        var lines: [String] = []
        while let newline = partial.firstIndex(of: 0x0A) {
            let chunk = partial[partial.startIndex..<newline]
            partial.removeSubrange(partial.startIndex...newline)
            lines.append(String(decoding: chunk, as: UTF8.self))
        }
        if partial.count > 100_000 { partial = Data() }
        return lines
    }
}
