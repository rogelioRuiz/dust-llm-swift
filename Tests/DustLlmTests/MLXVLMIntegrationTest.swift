import XCTest
@testable import DustLlm
import DustCore

/// Integration test for the MLX VLM pipeline.
/// Requires the Qwen3.5-2B-8bit model to be present in the HuggingFace cache.
/// Run with:
///   swift test --filter MLXVLMIntegrationTest
final class MLXVLMIntegrationTest: XCTestCase {

    private func modelPath() throws -> String {
        // Check common cache locations
        let candidates = [
            NSHomeDirectory() + "/Library/Caches/models/mlx-community/Qwen3.5-2B-8bit",
            NSHomeDirectory() + "/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-8bit/snapshots",
        ]

        for candidate in candidates {
            let configPath = candidate + "/config.json"
            if FileManager.default.fileExists(atPath: configPath) {
                return candidate
            }
        }

        // Check if env var is set
        if let envPath = ProcessInfo.processInfo.environment["MLX_MODEL_PATH"] {
            return envPath
        }

        throw XCTSkip("Qwen3.5-2B-8bit model not found. Set MLX_MODEL_PATH env var.")
    }

    /// A 64x64 red PNG image (VLM processor requires at least 32x32)
    private func redPixelPNG() -> Data {
        Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x40, 0x08, 0x02, 0x00, 0x00, 0x00, 0x25, 0x0B, 0xE6, 0x89, 0x00, 0x00, 0x00, 0x7C, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0xD5, 0xCE, 0x41, 0x11, 0x00, 0x30, 0x08, 0xC0, 0xB0, 0xAE, 0xFE, 0x3D, 0x33, 0x11, 0x3C, 0xB8, 0x46, 0x41, 0xDE, 0xD0, 0x26, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0x27, 0x71, 0x12, 0xE7, 0x75, 0x60, 0xEB, 0x03, 0x52, 0x85, 0x01, 0x7F, 0x7D, 0x14, 0x8B, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82])
    }

    func testMLXEngineLoadsAsVLM() throws {
        #if !canImport(MLXLLM)
        throw XCTSkip("MLXLLM not available on this platform")
        #else
        let path = try modelPath()
        print("=== Loading model from: \(path)")

        let engine = try MLXEngine(path: path, config: LLMConfig())

        print("=== metadata.hasVision = \(engine.metadata.hasVision)")
        print("=== supportsNativeImage = \(engine.supportsNativeImage)")
        print("=== contextSize = \(engine.nCtx)")

        XCTAssertTrue(engine.metadata.hasVision, "Model should load as VLM with hasVision=true")
        XCTAssertTrue(engine.supportsNativeImage, "supportsNativeImage should be true for VLM")
        #endif
    }

    func testMLXVLMGeneratesWithImage() throws {
        #if !canImport(MLXLLM)
        throw XCTSkip("MLXLLM not available on this platform")
        #else
        let path = try modelPath()
        print("=== Loading model from: \(path)")

        let engine = try MLXEngine(path: path, config: LLMConfig())

        guard engine.supportsNativeImage else {
            XCTFail("Model loaded but supportsNativeImage is false — VLMModelFactory likely failed")
            return
        }

        let messages: [[String: String]] = [
            ["role": "user", "content": "What color is this image?"]
        ]

        print("=== Calling generateWithNativeImage...")
        let result = try engine.generateWithNativeImage(
            messages: messages,
            imageData: redPixelPNG(),
            maxTokens: 50,
            sampler: SamplerConfig()
        )

        let text = try engine.detokenize(tokens: result.tokens)
        print("=== Generated text: \(text)")
        print("=== Token count: \(result.tokens.count)")
        print("=== Stop reason: \(result.stopReason)")

        XCTAssertGreaterThan(result.tokens.count, 0, "Should generate at least 1 token")
        XCTAssertFalse(text.isEmpty, "Generated text should not be empty")
        #endif
    }

    func testMLXVLMStreamGeneratesWithImage() throws {
        #if !canImport(MLXLLM)
        throw XCTSkip("MLXLLM not available on this platform")
        #else
        let path = try modelPath()
        let engine = try MLXEngine(path: path, config: LLMConfig())

        guard engine.supportsNativeImage else {
            XCTFail("Model loaded but supportsNativeImage is false")
            return
        }

        let messages: [[String: String]] = [
            ["role": "user", "content": "Describe this image briefly."]
        ]

        var tokens: [Int32] = []
        print("=== Calling generateStreamingWithNativeImage...")
        let stopReason = try engine.generateStreamingWithNativeImage(
            messages: messages,
            imageData: redPixelPNG(),
            maxTokens: 50,
            sampler: SamplerConfig(),
            isCancelled: { false },
            onToken: { token in
                tokens.append(token)
            }
        )

        let text = try engine.detokenize(tokens: tokens)
        print("=== Streamed text: \(text)")
        print("=== Token count: \(tokens.count)")
        print("=== Stop reason: \(stopReason)")

        XCTAssertGreaterThan(tokens.count, 0, "Should stream at least 1 token")
        #endif
    }
}
