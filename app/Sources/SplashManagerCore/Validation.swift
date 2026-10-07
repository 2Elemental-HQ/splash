import Foundation

/// Every value that reaches the `splash serve` command line passes one of
/// these checks. The management API accepts configuration ids only, never
/// free arguments, so these guard the values a person types into the app.
public enum Validation {
    public struct Failure: Error, Equatable, CustomStringConvertible {
        public let field: String
        public let message: String
        public var description: String { "\(field): \(message)" }
    }

    private static func matches(_ value: String, _ pattern: String) -> Bool {
        value.range(of: pattern, options: .regularExpression) != nil
    }

    /// `OWNER/REPO` or a GGUF's `OWNER/REPO:VARIANT`.
    public static func modelId(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let part = "[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
        guard matches(trimmed, "^\(part)/\(part)(:[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$") else {
            throw Failure(field: "model", message: "use OWNER/REPO or OWNER/REPO:VARIANT")
        }
        return trimmed
    }

    /// A branch, tag or commit. Never starts with `-`, so it cannot be an option.
    public static func revision(_ value: String?) throws -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        guard matches(value, "^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"), !value.contains("..") else {
            throw Failure(field: "revision", message: "use a branch, tag or commit name")
        }
        return value
    }

    public static func size(_ value: String?, field: String) throws -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        guard matches(value, "^(auto|[0-9]{1,12}[KkMmGg]?)$") else {
            throw Failure(field: field, message: "use auto, a number of bytes, or a number with K, M or G")
        }
        return value
    }

    public static func duration(_ value: String?, field: String) throws -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        guard matches(value, "^(off|[0-9]{1,6}(\\.[0-9]{1,3})?[smh]?)$") else {
            throw Failure(field: field, message: "use off, or seconds, or a number with s, m or h")
        }
        return value
    }

    public static func port(_ value: Int, field: String) throws -> Int {
        guard (1024...65535).contains(value) else {
            throw Failure(field: field, message: "use a port from 1024 to 65535")
        }
        return value
    }

    public static func hostName(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard matches(trimmed, "^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$") else {
            throw Failure(field: "allowed host", message: "use a host name such as mymac.tailnet.ts.net")
        }
        return trimmed
    }

    public static func configName(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 80,
              !trimmed.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
        else { throw Failure(field: "name", message: "use 1 to 80 printable characters") }
        return trimmed
    }
}
