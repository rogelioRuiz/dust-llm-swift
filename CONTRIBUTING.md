# Contributing to dust-llm-swift

Thanks for your interest in contributing! This guide will help you get set up and understand our development workflow.

## Prerequisites

- **macOS** with Swift 5.9+ toolchain
- **Git**
- **dust-core-swift** cloned as a sibling directory (`../dust-core-swift`)

The llama.cpp native code is included as a Git submodule under `native/`.

## Getting Started

```bash
# Clone both repos side-by-side
git clone https://github.com/rogelioRuiz/dust-core-swift.git
git clone --recursive https://github.com/rogelioRuiz/dust-llm-swift.git

cd dust-llm-swift

# Build
swift build

# Run tests
swift test
```

## Project Structure

```
Sources/DustLlm/
  ChatTemplateEngine.swift   # Chat prompt formatting
  LlamaContext.swift         # Native context lifecycle wrapper
  LlamaEngine.swift         # llama.cpp inference engine
  LlamaSession.swift        # Single inference session
  LLMConfig.swift           # Model configuration
  LLMSessionManager.swift   # Session lifecycle and caching
  VisionEncoder.swift       # Multimodal vision input encoding

native/                      # llama.cpp submodule

Tests/DustLlmTests/
  LLMChatTemplateTests.swift    # chat template tests
  LLMGenerationTests.swift      # generation tests
  LLMRegistryTests.swift        # registry tests
  LLMSessionManagerTests.swift  # session manager tests
  LLMStreamingTests.swift       # streaming tests
  LLMVisionTests.swift          # vision tests (2 skipped)
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feat/my-feature
```

### 2. Make your changes

- Follow existing Swift conventions in the codebase
- Use `async throws` for I/O-bound operations
- Add tests for new functionality

### 3. Add the license header

All `.swift` files must include the Apache 2.0 header:

```swift
//
// Copyright 2026 Rogelio Ruiz Perez
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
```

### 4. Run checks

```bash
swift test          # All 50 tests must pass (2 vision tests skipped)
swift build         # Clean build
```

### 5. Commit with a conventional message

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add LoRA adapter loading support
fix: correct KV cache eviction under memory pressure
docs: update README usage examples
chore(deps): bump dust-core-swift to 0.2.0
```

### 6. Open a pull request

Push your branch and open a PR against `main`.

## Reporting Issues

- **Bugs**: Open an issue with steps to reproduce
- **Features**: Open an issue describing the use case and proposed API

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and constructive.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
