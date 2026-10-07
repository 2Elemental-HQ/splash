import AppKit
import SplashManagerCore

/// Start, stop and switch with the dialogs a person needs: busy calls, downloads, errors.
@MainActor
enum StartFlow {
    static func run(_ supervisor: SplashSupervisor, configId: String?) async {
        guard let configId else { return }
        do {
            _ = try await supervisor.start(configId: configId, allowDownload: false)
        } catch let error as AppError where error.code == "download_required" {
            await confirmDownload(supervisor, configId: configId, error: error) {
                _ = try await supervisor.start(configId: configId, allowDownload: true)
            }
        } catch let error as AppError { Dialogs.show("Splash did not start", error.message) }
        catch { Dialogs.show("Splash did not start", "\(error)") }
    }

    static func confirmDownload(_ supervisor: SplashSupervisor, configId: String, error: AppError,
                                action: @escaping () async throws -> Void) async {
        guard let config = supervisor.store.config(id: configId) else { return }
        let assessment = supervisor.assess(config)
        let estimates = await DownloadEstimator.estimate(modelId: config.modelId, revision: config.revision, missing: assessment.missing)
        let known = estimates.compactMap(\.bytes).reduce(0, +)
        var text = "Starting \(config.displayName) downloads files from Hugging Face:\n"
        for estimate in estimates {
            text += "\n• \(estimate.repository): \(estimate.bytes.map(formatBytes) ?? "size unknown (\(estimate.error ?? "no data"))")"
        }
        if assessment.missing.contains("(draft unknown)") { text += "\n• the matching draft model: unknown to this app" }
        if known > 0 { text += "\n\nAbout \(formatBytes(known)) in total." }
        if Dialogs.confirm("Download needed", text, ok: "Download and Start") {
            do { try await action() } catch let error as AppError { Dialogs.show("Splash did not start", error.message) } catch {}
        }
    }
}

@MainActor
enum StopFlow {
    /// Drains: no new calls, running calls finish, then Splash exits. A Splash that cannot drain,
    /// or a person in a hurry, goes through an explicit confirmation.
    static func run(_ supervisor: SplashSupervisor) async {
        do { _ = try await supervisor.stop(force: false) }
        catch let error as AppError where error.code == "drain_unsupported" {
            if Dialogs.confirm("Stop without draining?",
                               "\(error.message)\n\nCalls that are running or that arrive now can be cut.", ok: "Stop Now") {
                _ = try? await supervisor.stop(force: true)
            }
        } catch let error as AppError { Dialogs.show("Splash did not stop", error.message) }
        catch { Dialogs.show("Splash did not stop", "\(error)") }
    }

    /// An explicit choice that ends running calls.
    static func now(_ supervisor: SplashSupervisor) async {
        let active = supervisor.activity?.activeRequests ?? 0
        let text = active > 0 ? "\(active) request(s) are running. Stopping now ends them." : "Splash stops at once."
        if Dialogs.confirm("Stop now?", text, ok: "Stop Now") { _ = try? await supervisor.stop(force: true) }
    }
}

@MainActor
enum SwitchFlow {
    static func run(_ supervisor: SplashSupervisor, configId: String) async {
        do { _ = try await supervisor.switchTo(configId: configId, allowDownload: false) }
        catch let error as AppError where error.code == "download_required" {
            await StartFlow.confirmDownload(supervisor, configId: configId, error: error) {
                _ = try await supervisor.switchTo(configId: configId, allowDownload: true)
            }
        } catch let error as AppError where error.code == "drain_unsupported" {
            if Dialogs.confirm("Switch without draining?",
                               "\(error.message)\n\nCalls that are running or that arrive now can be cut.", ok: "Switch Now") {
                do { _ = try await supervisor.switchTo(configId: configId, allowDownload: true, force: true) }
                catch let error as AppError { Dialogs.show("Model not switched", error.message) } catch {}
            }
        } catch let error as AppError { Dialogs.show("Model not switched", error.message) }
        catch { Dialogs.show("Model not switched", "\(error)") }
    }
}

@MainActor
enum Dialogs {
    static func show(_ title: String, _ text: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = text
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    static func confirm(_ title: String, _ text: String, ok: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = text
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: ok)
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertSecondButtonReturn
    }
}
