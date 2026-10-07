// swift-tools-version:6.0
import PackageDescription

// Splash Manager: a menu bar app that supervises the installed `splash serve`
// process. It lives beside the upstream tree and shares no files with it.
let package = Package(
    name: "SplashManager",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SplashManager", targets: ["SplashManager"]),
        .library(name: "SplashManagerCore", targets: ["SplashManagerCore"]),
    ],
    targets: [
        .target(name: "SplashManagerCore"),
        .executableTarget(name: "SplashManager", dependencies: ["SplashManagerCore"]),
        .testTarget(name: "SplashManagerCoreTests", dependencies: ["SplashManagerCore"]),
    ],
    swiftLanguageModes: [.v5]
)
