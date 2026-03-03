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
        .package(url: "https://github.com/rogelioRuiz/dust-core-swift.git", from: "0.1.0"),
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
                "llama.cpp/licenses",
                "llama.cpp/media",
                "llama.cpp/models",
                "llama.cpp/pocs",
                "llama.cpp/requirements",
                "llama.cpp/scripts",
                "llama.cpp/tests",
                "llama.cpp/tools",
                "llama.cpp/vendor",
                // CMakeLists.txt / cmake files (not compilable by SPM)
                "llama.cpp/src/CMakeLists.txt",
                "llama.cpp/ggml/src/CMakeLists.txt",
                "llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt",
                "llama.cpp/ggml/src/ggml-cpu/cmake",
                // Platform-specific CPU backends not relevant for Apple
                "llama.cpp/ggml/src/ggml-cpu/spacemit",
                "llama.cpp/ggml/src/ggml-cpu/kleidiai",
                "llama.cpp/ggml/src/ggml-cpu/arch/loongarch",
                "llama.cpp/ggml/src/ggml-cpu/arch/powerpc",
                "llama.cpp/ggml/src/ggml-cpu/arch/riscv",
                "llama.cpp/ggml/src/ggml-cpu/arch/s390",
                "llama.cpp/ggml/src/ggml-cpu/arch/wasm",
                "llama.cpp/ggml/src/ggml-cpu/arch/x86",
                "llama.cpp/ggml/src/ggml-metal/CMakeLists.txt",
                // Metal shader — excluded; embedded via ggml-metal-embed instead
                "llama.cpp/ggml/src/ggml-metal/ggml-metal.metal",
                // Non-Metal GPU backends
                "llama.cpp/ggml/src/ggml-blas",
                "llama.cpp/ggml/src/ggml-cann",
                "llama.cpp/ggml/src/ggml-cuda",
                "llama.cpp/ggml/src/ggml-hexagon",
                "llama.cpp/ggml/src/ggml-hip",
                "llama.cpp/ggml/src/ggml-musa",
                "llama.cpp/ggml/src/ggml-opencl",
                "llama.cpp/ggml/src/ggml-rpc",
                "llama.cpp/ggml/src/ggml-sycl",
                "llama.cpp/ggml/src/ggml-virtgpu",
                "llama.cpp/ggml/src/ggml-vulkan",
                "llama.cpp/ggml/src/ggml-webgpu",
                "llama.cpp/ggml/src/ggml-zdnn",
                "llama.cpp/ggml/src/ggml-zendnn",
                // Embedded Metal shader data — consumed by .incbin in .s, not compiled
                "ggml-metal-embed/ggml-metal-embed.metaldata",
            ],
            sources: [
                "llama.cpp/src",
                "llama.cpp/ggml/src",
                "ggml-metal-embed",
            ],
            publicHeadersPath: "llama-spm-headers",
            cSettings: [
                .headerSearchPath("llama.cpp/include"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/src"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/ggml/src/ggml-cpu"),
                .headerSearchPath("ggml-metal-embed"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_CPU"),
                .define("GGML_METAL_EMBED_LIBRARY"),
                .define("GGML_VERSION", to: "\"0.9.7\""),
                .define("GGML_COMMIT", to: "\"b8189\""),
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
                .define("GGML_METAL_EMBED_LIBRARY"),
                .define("GGML_VERSION", to: "\"0.9.7\""),
                .define("GGML_COMMIT", to: "\"b8189\""),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "mtmd",
            dependencies: ["llama"],
            path: "native",
            exclude: [
                "llama.cpp/tools/mtmd/deprecation-warning.cpp",
                "llama.cpp/tools/mtmd/mtmd-cli.cpp",
                "llama.cpp/tools/mtmd/legacy-models",
                "llama.cpp/tools/mtmd/CMakeLists.txt",
                "llama.cpp/tools/mtmd/tests.sh",
                "llama.cpp/tools/mtmd/requirements.txt",
                "llama.cpp/tools/mtmd/test-1.jpeg",
                "llama.cpp/tools/mtmd/test-2.mp3",
                "llama.cpp/tools/mtmd/README.md",
                // Prevent Xcode from auto-compiling Metal shader (embedded in llama target)
                "llama.cpp/ggml/src/ggml-metal/ggml-metal.metal",
            ],
            sources: [
                "llama.cpp/tools/mtmd/clip.cpp",
                "llama.cpp/tools/mtmd/mtmd.cpp",
                "llama.cpp/tools/mtmd/mtmd-helper.cpp",
                "llama.cpp/tools/mtmd/mtmd-audio.cpp",
                "llama.cpp/tools/mtmd/models",
            ],
            publicHeadersPath: "mtmd-spm-headers",
            cxxSettings: [
                .headerSearchPath("llama.cpp/tools/mtmd"),
                .headerSearchPath("llama.cpp/include"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/vendor"),
                .unsafeFlags(["-Wno-cast-qual", "-UDEBUG"]),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "DustLlm",
            dependencies: [
                "llama",
                "mtmd",
                .product(name: "DustCore", package: "dust-core-swift"),
            ],
            path: "Sources/DustLlm"
        ),
        .testTarget(
            name: "DustLlmTests",
            dependencies: ["DustLlm", "mtmd"],
            path: "Tests/DustLlmTests",
            resources: [
                .copy("Fixtures/tiny-test.gguf"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5],
    cxxLanguageStandard: .cxx17
)
