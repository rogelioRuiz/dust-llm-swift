import XCTest
@testable import DustLlm
import DustCore

private final class Box<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

final class LLMVisionTests: XCTestCase {
    func testL6T1LoadVisionCapableModelInitializesVisionEncoder() throws {
        let fixtureURL = try fixtureURL()
        let visionFactoryCalled = Box(false)
        let mockEncoder = MockVisionEncoder()

        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, config, priority, _ in
                let metadata = LLMModelMetadata(name: "vision-model", chatTemplate: nil, hasVision: true)
                let encoder: VisionEncoderProtocol?
                if metadata.hasVision {
                    visionFactoryCalled.value = true
                    encoder = mockEncoder
                } else {
                    encoder = nil
                }
                return LlamaSession(
                    sessionId: modelId,
                    engine: VisionMockLlamaEngine(),
                    metadata: metadata,
                    priority: priority,
                    visionEncoder: encoder
                )
            }
        )

        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "vision-model",
            config: LLMConfig(),
            priority: .interactive
        )

        XCTAssertTrue(session.metadata.hasVision)
        XCTAssertNotNil(session.visionEncoder)
        XCTAssertTrue(visionFactoryCalled.value)
    }

    func testL6T2TextOnlyModelKeepsVisionEncoderNil() {
        let session = LlamaSession(
            sessionId: "text-only",
            engine: VisionMockLlamaEngine(),
            metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: false),
            priority: .interactive
        )

        XCTAssertFalse(session.metadata.hasVision)
        XCTAssertNil(session.visionEncoder)
    }

    func testL6T3ImageToTextOnlyModelThrowsUnsupportedOperation() throws {
        let session = LlamaSession(
            sessionId: "text-only",
            engine: VisionMockLlamaEngine(),
            metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: false),
            priority: .interactive
        )

        XCTAssertThrowsError(
            try session.generate(
                prompt: "describe",
                imageData: pngData(),
                maxTokens: 2,
                stopSequences: [],
                sampler: SamplerConfig()
            )
        ) { error in
            guard case .unsupportedOperation = error as? LlamaError else {
                return XCTFail("Expected unsupportedOperation, got \(error)")
            }
        }
    }

    func testL6T4RealClipEncodeReturnsEmbeddingWhenEnvIsSet() throws {
        guard let mmprojPath = ProcessInfo.processInfo.environment["LLAVA_MMPROJ_PATH"],
              let modelPath = ProcessInfo.processInfo.environment["LLAVA_MODEL_PATH"] else {
            throw XCTSkip("LLAVA_MMPROJ_PATH and LLAVA_MODEL_PATH must both be set")
        }

        let context = try LlamaContext(path: modelPath, config: LLMConfig())
        guard let model = context.model else {
            throw XCTSkip("Failed to load model")
        }

        let encoder = try VisionEncoder(mmprojPath: mmprojPath, model: model)
        defer { encoder.close() }

        let embedding = try encoder.encode(imageBytes: pngData())
        defer { encoder.freeEmbedding(embedding) }

        XCTAssertGreaterThan(embedding.tokenCount, 0)
    }

    func testL6T5VisionEncoderFailurePropagatesAsInferenceError() throws {
        let engine = VisionMockLlamaEngine()
        let visionEncoder = MockVisionEncoder(encodeError: DustCoreError.inferenceFailed(detail: "Failed to encode image"))
        let session = makeSession(engine: engine, visionEncoder: visionEncoder)

        XCTAssertThrowsError(
            try session.generate(
                prompt: "describe",
                imageData: Data("not-an-image".utf8),
                maxTokens: 1,
                stopSequences: [],
                sampler: SamplerConfig()
            )
        ) { error in
            XCTAssertEqual(error as? DustCoreError, .inferenceFailed(detail: "Failed to encode image"))
        }
    }

    func testL6T6ImageEmbeddingIsInjectedAfterPromptTokens() throws {
        let engine = VisionMockLlamaEngine()
        let visionEncoder = MockVisionEncoder(imageTokenCount: 32)
        let session = makeSession(engine: engine, visionEncoder: visionEncoder)

        _ = try session.generate(
            prompt: "describe",
            imageData: pngData(),
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(engine.lastVisionPromptTokens, [11, 12, 13])
        XCTAssertEqual(visionEncoder.lastEvalNPastBefore, 3)
        XCTAssertEqual(visionEncoder.lastEvalNPastAfter, 35)
    }

    func testL6T7OversizedImageDoesNotCrashWithMockEncoder() throws {
        let engine = VisionMockLlamaEngine()
        let visionEncoder = MockVisionEncoder(imageTokenCount: 64)
        let session = makeSession(engine: engine, visionEncoder: visionEncoder)
        let oversized = Data(repeating: 0xFF, count: 64 * 1024)

        let result = try session.generate(
            prompt: "describe",
            imageData: oversized,
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "ok")
        XCTAssertEqual(result.tokenCount, 2)
    }

    func testL6T8StreamGenerateWithImageReportsPromptTokens() throws {
        guard let mmprojPath = ProcessInfo.processInfo.environment["LLAVA_MMPROJ_PATH"],
              let modelPath = ProcessInfo.processInfo.environment["LLAVA_MODEL_PATH"] else {
            throw XCTSkip("LLAVA_MMPROJ_PATH and LLAVA_MODEL_PATH must both be set")
        }

        let engine = VisionMockLlamaEngine()
        engine.shouldCallVisionEval = false
        let context = try LlamaContext(path: modelPath, config: LLMConfig())
        guard let model = context.model else {
            throw XCTSkip("Failed to load model")
        }
        let visionEncoder = try VisionEncoder(mmprojPath: mmprojPath, model: model)
        defer { visionEncoder.close() }
        let session = makeSession(engine: engine, visionEncoder: visionEncoder)
        let completeExpectation = expectation(description: "complete")
        let observedPromptTokens = Box(0)
        let observedTokenCount = Box(0)

        session.streamGenerate(
            prompt: "describe",
            imageData: pngData(),
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in },
            onComplete: { _, tokenCount, promptTokens, _, _ in
                observedTokenCount.value = tokenCount
                observedPromptTokens.value = promptTokens
                completeExpectation.fulfill()
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
                completeExpectation.fulfill()
            }
        )

        wait(for: [completeExpectation], timeout: 2.0)
        XCTAssertEqual(observedTokenCount.value, 2)
        XCTAssertGreaterThan(observedPromptTokens.value, 3)
    }

    private func makeSession(
        engine: VisionMockLlamaEngine,
        visionEncoder: VisionEncoderProtocol
    ) -> LlamaSession {
        engine.tokenizeResult = [11, 12, 13]
        engine.detokenizeMap = [
            [101]: "o",
            [101, 102]: "ok",
        ]
        engine.generateResult = ([101, 102], .maxTokens)
        engine.streamingTokens = [101, 102]

        return LlamaSession(
            sessionId: "vision-session",
            engine: engine,
            metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: true),
            priority: .interactive,
            visionEncoder: visionEncoder
        )
    }

    private func fixtureURL() throws -> URL {
        if let bundled = Bundle.module.url(forResource: "tiny-test", withExtension: "gguf") {
            return bundled
        }

        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/tiny-test.gguf")

        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("tiny-test.gguf fixture missing")
        }

        return fallback
    }

    private func pngData() -> Data {
        Data([
            137, 80, 78, 71, 13, 10, 26, 10,
            0, 0, 0, 13, 73, 72, 68, 82,
            0, 0, 0, 1, 0, 0, 0, 1,
            8, 4, 0, 0, 0, 181, 28, 12,
            2, 0, 0, 0, 11, 73, 68, 65,
            84, 120, 218, 99, 252, 255, 31, 0,
            3, 3, 2, 0, 239, 166, 229, 177,
            0, 0, 0, 0, 73, 69, 78, 68,
            174, 66, 96, 130,
        ])
    }
}

private final class VisionMockLlamaEngine: LlamaEngine, @unchecked Sendable {
    var nCtx: UInt32 = 2048
    var tokenizeResult: [Int32] = []
    var detokenizeMap: [[Int32]: String] = [:]
    var generateResult: (tokens: [Int32], stopReason: StopReason) = ([], .maxTokens)
    var streamingTokens: [Int32] = []
    var shouldCallVisionEval = true
    var lastVisionPromptTokens: [Int32]?

    func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        _ = text
        _ = addSpecial
        return tokenizeResult
    }

    func detokenize(tokens: [Int32]) throws -> String {
        detokenizeMap[tokens] ?? ""
    }

    func vocabEosToken() -> Int32 {
        0
    }

    func isEog(token: Int32) -> Bool {
        _ = token
        return false
    }

    func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        _ = promptTokens
        _ = maxTokens
        _ = sampler
        return generateResult
    }

    func generateStreaming(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        _ = promptTokens
        _ = maxTokens
        _ = sampler

        for token in streamingTokens where !isCancelled() {
            onToken(token)
        }

        return .maxTokens
    }

    func generateWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        _ = maxTokens
        _ = sampler
        lastVisionPromptTokens = promptTokens

        if shouldCallVisionEval {
            var nPast = Int32(promptTokens.count)
            try visionEncoder.evalImageEmbed(
                embedding: imageEmbedding,
                context: OpaquePointer(bitPattern: 0x1)!,
                nPast: &nPast
            )
        }

        return generateResult
    }

    func generateStreamingWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        _ = try generateWithVision(
            promptTokens: promptTokens,
            imageEmbedding: imageEmbedding,
            visionEncoder: visionEncoder,
            maxTokens: maxTokens,
            sampler: sampler
        )

        for token in streamingTokens where !isCancelled() {
            onToken(token)
        }

        return .maxTokens
    }
}

private final class MockVisionEncoder: VisionEncoderProtocol {
    let imageTokenCount: Int
    var encodeError: Error?
    var lastEvalNPastBefore: Int?
    var lastEvalNPastAfter: Int?

    init(
        imageTokenCount: Int = 576,
        encodeError: Error? = nil
    ) {
        self.imageTokenCount = imageTokenCount
        self.encodeError = encodeError
    }

    func encode(imageBytes: Data) throws -> ImageEmbedding {
        _ = imageBytes
        if let encodeError {
            throw encodeError
        }

        // Use a dummy opaque pointer for testing — never dereferenced by mock eval.
        let dummyChunks = OpaquePointer(bitPattern: 0x1)!
        return ImageEmbedding(chunks: dummyChunks, tokenCount: imageTokenCount)
    }

    func evalImageEmbed(
        embedding: ImageEmbedding,
        context: OpaquePointer,
        nPast: inout Int32
    ) throws {
        _ = context
        lastEvalNPastBefore = Int(nPast)
        nPast += Int32(embedding.tokenCount)
        lastEvalNPastAfter = Int(nPast)
    }

    func freeEmbedding(_ embedding: ImageEmbedding) {
        _ = embedding
    }

    func close() {}
}
