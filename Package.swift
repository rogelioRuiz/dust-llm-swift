// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "dust-llm-swift",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(
            name: "DustLlm",
            targets: ["DustLlm"]
        )
    ],
    dependencies: [
        .package(name: "dust-core-swift", path: "../dust-core-swift"),
    ],
    targets: [
        .target(
            name: "llama",
            path: "native",
            exclude: [
                "llama.cpp/.devops",
                "llama.cpp/.gemini",
                "llama.cpp/.github",
                "llama.cpp/benches",
                "llama.cpp/ci",
                "llama.cpp/cmake",
                "llama.cpp/common",
                "llama.cpp/docs",
                "llama.cpp/examples",
                "llama.cpp/gguf-py",
                "llama.cpp/grammars",
                "llama.cpp/models",
                "llama.cpp/pocs",
                "llama.cpp/prompts",
                "llama.cpp/scripts",
                "llama.cpp/tests",
                "llama.cpp/tools",
                // CMakeLists.txt / cmake files (not compilable by SPM)
                "llama.cpp/src/CMakeLists.txt",
                "llama.cpp/ggml/src/CMakeLists.txt",
                "llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt",
                "llama.cpp/ggml/src/ggml-cpu/cmake",
                "llama.cpp/ggml/src/ggml-metal/CMakeLists.txt",
                // Metal shader — excluded from sources, added as resource below
                "llama.cpp/ggml/src/ggml-metal/ggml-metal.metal",
                // Non-Metal GPU backends
                "llama.cpp/ggml/src/ggml-blas",
                "llama.cpp/ggml/src/ggml-cann",
                "llama.cpp/ggml/src/ggml-cuda",
                "llama.cpp/ggml/src/ggml-hip",
                "llama.cpp/ggml/src/ggml-kompute",
                "llama.cpp/ggml/src/ggml-musa",
                "llama.cpp/ggml/src/ggml-opencl",
                "llama.cpp/ggml/src/ggml-rpc",
                "llama.cpp/ggml/src/ggml-sycl",
                "llama.cpp/ggml/src/ggml-vulkan",
            ],
            sources: [
                "llama.cpp/src",
                "llama.cpp/ggml/src",
            ],
            resources: [
                .copy("llama.cpp/ggml/src/ggml-metal/ggml-metal.metal"),
            ],
            publicHeadersPath: "llama-spm-headers",
            cSettings: [
                .headerSearchPath("llama.cpp/include"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/src"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/ggml/src/ggml-cpu"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_CPU"),
                // SPM defines SWIFT_PACKAGE but doesn't generate SWIFTPM_MODULE_BUNDLE for C targets.
                // ggml-metal.m uses it to find the Metal shader at runtime — fall back to mainBundle.
                .define("SWIFTPM_MODULE_BUNDLE", to: "[NSBundle mainBundle]"),
                .unsafeFlags(["-fno-objc-arc"]),
            ],
            cxxSettings: [
                .headerSearchPath("llama.cpp/include"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/src"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/ggml/src/ggml-cpu"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_CPU"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "llava",
            dependencies: ["llama"],
            path: "native",
            sources: [
                "llama.cpp/examples/llava/clip.cpp",
                "llama.cpp/examples/llava/llava.cpp",
            ],
            publicHeadersPath: "llava-spm-headers",
            cxxSettings: [
                .headerSearchPath("llama.cpp/examples/llava"),
                .headerSearchPath("llama.cpp/include"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/common"),
                .unsafeFlags(["-Wno-cast-qual"]),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "DustLlm",
            dependencies: [
                "llama",
                "llava",
                .product(name: "DustCore", package: "dust-core-swift"),
            ],
            path: "Sources/DustLlm"
        ),
        .testTarget(
            name: "DustLlmTests",
            dependencies: ["DustLlm", "llava"],
            path: "Tests/DustLlmTests",
            resources: [
                .copy("Fixtures/tiny-test.gguf"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5],
    cxxLanguageStandard: .cxx17
)
