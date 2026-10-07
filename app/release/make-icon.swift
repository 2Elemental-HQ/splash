// Draws the app icon: a bolt on a rounded square. Run once; the .icns is committed.
//   swift release/make-icon.swift release/AppIcon.iconset && iconutil -c icns release/AppIcon.iconset -o release/AppIcon.icns
import AppKit

let output = CommandLine.arguments.dropFirst().first ?? "AppIcon.iconset"
try? FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)

func render(_ size: Int) -> Data {
    let image = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
        let inset = rect.width * 0.08
        let body = rect.insetBy(dx: inset, dy: inset)
        let path = NSBezierPath(roundedRect: body, xRadius: body.width * 0.225, yRadius: body.width * 0.225)
        NSGradient(colors: [NSColor(red: 0.13, green: 0.35, blue: 0.78, alpha: 1), NSColor(red: 0.30, green: 0.78, blue: 0.86, alpha: 1)])?
            .draw(in: path, angle: -60)
        let config = NSImage.SymbolConfiguration(pointSize: rect.width * 0.5, weight: .bold)
        if let bolt = NSImage(systemSymbolName: "bolt.fill", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
            let tinted = NSImage(size: bolt.size, flipped: false) { r in
                bolt.draw(in: r); NSColor.white.set(); r.fill(using: .sourceAtop); return true
            }
            let target = NSRect(x: rect.midX - tinted.size.width / 2, y: rect.midY - tinted.size.height / 2,
                                width: tinted.size.width, height: tinted.size.height)
            tinted.draw(in: target)
        }
        return true
    }
    let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
    return rep.representation(using: .png, properties: [:])!
}

for base in [16, 32, 128, 256, 512] {
    try! render(base).write(to: URL(fileURLWithPath: "\(output)/icon_\(base)x\(base).png"))
    try! render(base * 2).write(to: URL(fileURLWithPath: "\(output)/icon_\(base)x\(base)@2x.png"))
}
