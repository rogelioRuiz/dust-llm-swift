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
  <img alt="MLX" src="https://img.shields.io/badge/MLX-Apple_Silicon-green">
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

Dual-backend LLM inference for Dust — GGUF via llama.cpp and MLX for Apple Silicon. iOS/macOS with Metal GPU acceleration.

**Version: 0.1.0**

## Overview

Provides a Swift-native API for running large language models on-device with two backends:

- **GGUF/llama.cpp** — Works on all iOS 16+ / macOS 14+ devices. CPU + Metal GPU acceleration.
- **MLX** — Optimized for Apple Silicon (iPhone, iPad, Mac). Requires iOS 17+ / macOS 14+ with Metal GPU. Uses [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) for model loading and inference.

The backend is selected automatically based on model format: directories containing `config.json` + `.safetensors` files use MLX; `.gguf` files use llama.cpp. Builds on [dust-core-swift](../dust-core-swift).

```
dust-llm-swift/
├── Package.swift                           # SPM: product "DustLlm", iOS 16+ / macOS 14+
├── DustLlm.podspec                        # CocoaPods spec (module name: DustLlm)
├── native/llama.cpp/                       # llama.cpp submodule (clone with --recursive)
├── Sources/DustLlm/
│   ├── LlamaEngine.swift                   # Protocol for inference backends
│   ├── LlamaContext.swift                  # GGUF/llama.cpp backend
│   ├── MLXEngine.swift                     # MLX backend (Apple Silicon)
│   ├── MLXModelDetector.swift              # Auto-detects MLX model directories
│   ├── LLMSessionManager.swift             # Dual-backend routing + session cache
│   ├── LlamaSession.swift                  # Unified session (works with either backend)
│   ├── ChatTemplateEngine.swift            # Jinja2 subset renderer
│   └── VisionEncoder.swift                 # CLIP/LLaVA multimodal
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
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (MLXLLM, MLXLMCommon — conditionally compiled via `#if canImport(MLXLLM)`)

## Usage

### GGUF model (llama.cpp)

```swift
import DustLlm

let manager = LLMSessionManager()
let session = try manager.loadModel(
    path: "/path/to/model.gguf",
    modelId: "qwen",
    config: LLMConfig(contextSize: 2048, nGpuLayers: -1),
    priority: .interactive
)
```

### MLX model (Apple Silicon)

```swift
import DustLlm

// MLX models are directories with config.json + .safetensors files
// Backend is selected automatically — same API as GGUF
let manager = LLMSessionManager()
let session = try manager.loadModel(
    path: "/path/to/Qwen3.5-2B-8bit/",  // directory, not a file
    modelId: "qwen-mlx",
    config: LLMConfig(contextSize: 2048),
    priority: .interactive
)
```

Both backends expose the same `LlamaSession` API for tokenization, generation, streaming, and chat.

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
