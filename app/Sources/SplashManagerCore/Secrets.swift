import Foundation
import Security
import CryptoKit

public protocol SecretStore: AnyObject, Sendable {
    func read(_ account: String) -> String?
    func write(_ value: String, account: String) throws
}

public enum SecretAccount {
    public static let managementToken = "management-token"
    public static let inferenceKey = "inference-api-key"
}

public final class KeychainStore: SecretStore, @unchecked Sendable {
    private let service: String
    public init(service: String = "net.2elemental.splash-manager") { self.service = service }

    public func read(_ account: String) -> String? {
        var result: AnyObject?
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    public func write(_ value: String, account: String) throws {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let data = Data(value.utf8)
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return }
        guard update == errSecItemNotFound else { throw AppError("keychain_failed", "Keychain error \(update)", status: 500) }
        var add = base
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let status = SecItemAdd(add as CFDictionary, nil)
        guard status == errSecSuccess else { throw AppError("keychain_failed", "Keychain error \(status)", status: 500) }
    }
}

public final class MemorySecretStore: SecretStore, @unchecked Sendable {
    private var values: [String: String] = [:]
    private let lock = NSLock()
    public init() {}
    public func read(_ account: String) -> String? { lock.lock(); defer { lock.unlock() }; return values[account] }
    public func write(_ value: String, account: String) throws { lock.lock(); values[account] = value; lock.unlock() }
}

public enum Secrets {
    /// 256 bits from the system generator, URL-safe.
    public static func generate() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        precondition(status == errSecSuccess, "random generator failed")
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    public static func ensure(_ account: String, in store: SecretStore) throws -> String {
        if let existing = store.read(account), !existing.isEmpty { return existing }
        let fresh = generate()
        try store.write(fresh, account: account)
        return fresh
    }

    /// Compares digests so the time taken does not depend on a shared prefix.
    public static func equal(_ a: String, _ b: String) -> Bool {
        let x = Data(SHA256.hash(data: Data(a.utf8))), y = Data(SHA256.hash(data: Data(b.utf8)))
        var diff: UInt8 = 0
        for (l, r) in zip(x, y) { diff |= l ^ r }
        return diff == 0
    }
}
