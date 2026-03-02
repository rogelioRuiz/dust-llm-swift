<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="SPM" src="https://img.shields.io/badge/SPM-DustLlm-F05138">
  <img alt="CocoaPods" src="https://img.shields.io/badge/CocoaPods-DustLlm-EE3322">
  <a href="https://swift.org"><img alt="Swift" src="https://img.shields.io/badge/Swift-5.9-orange.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/Platforms-iOS_16+_|_macOS_14+-lightgrey">
  <img alt="GGUF" src="https://img.shields.io/badge/GGUF-llama.cpp-blueviolet">
  <a href="https://github.com/rogelioRuiz/dust-llm-swift/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rogelioRuiz/dust-llm-swift/actions/workflows/ci.yml/badge.svg?branch=main"></a>
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<strong>dust-llm-swift</strong> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# dust-llm-swift

GGUF/llama.cpp inference and chat runtime for Dust — iOS/macOS with Metal GPU acceleration.

**Version: 0.1.0**

## Overview

Provides a Swift-native API for running GGUF large language models via llama.cpp with Metal acceleration. Builds on [dust-core-swift](../dust-core-swift). Requires iOS 16+ / macOS 14+ (Metal).

```
dust-llm-swift/
├── Package.swift                           # SPM: product "DustLlm", iOS 16+ / macOS 14+
├── DustLlm.podspec                        # CocoaPods spec (module name: DustLlm)
├── native/llama.cpp/                       # llama.cpp submodule (clone with --recursive)
├── Sources/DustLlm/
│   ├── LlamaEngine.swift
│   ├── ChatSession.swift
│   ├── GenerationSession.swift
│   ├── StreamingHandler.swift
│   ├── VisionProcessor.swift
│   └── LlmRegistry.swift
└── Tests/DustLlmTests/
    └── Fixtures/
        └── tiny-test.gguf                  # Minimal model for integration tests
```

> **Note:** llama.cpp is included as a git submodule. Clone with `--recursive` or run `git submodule update --init` after cloning.

> **Note:** First build compiles llama.cpp from source (~5 min).

## Install

### Swift Package Manager — local

```swift
// Package.swift
dependencies: [
    .package(name: "dust-llm-swift", path: "../dust-llm-swift"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DustLlm", package: "dust-llm-swift"),
        ]
    )
]
```

### Swift Package Manager — remote (when published)

```swift
.package(url: "https://github.com/rogelioRuiz/dust-llm-swift.git", from: "0.1.0")
```

### CocoaPods

```ruby
pod 'DustLlm', '~> 0.1'
```

## Dependencies

- [dust-core-swift](../dust-core-swift) (DustCore)

## Usage

```swift
import DustLlm

// 1. Load a GGUF model
let engine = try LlamaEngine(modelPath: modelURL, nThreads: 4)

// 2. Start a chat session
let chat = ChatSession(engine: engine, systemPrompt: "You are a helpful assistant.")
let reply = try await chat.send("What is 2 + 2?")

// 3. Or stream tokens
let gen = GenerationSession(engine: engine) { token in
    print(token, terminator: "")
}
try await gen.generate("Once upon a time")

// 4. Clean up
engine.close()
```

## Test

```bash
cd dust-llm-swift
swift test    # 50 XCTest tests
```

2 vision tests are skipped unless the `LLAVA_MMPROJ_PATH` environment variable is set. Tests use the bundled `tiny-test.gguf` fixture. Requires macOS with Swift toolchain — no Xcode project needed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
