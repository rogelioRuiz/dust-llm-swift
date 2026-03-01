<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

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

Copyright 2026 T6X. Licensed under the [Apache License 2.0](LICENSE).
