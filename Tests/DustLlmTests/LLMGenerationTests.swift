import XCTest
@testable import DustLlm
import DustCore

final class LLMGenerationTests: XCTestCase {
    func testL2T1TokenizeKnownStringReturnsExpectedTokens() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [10, 20, 30, 40, 50]
        let session = makeSession(engine: engine)

        let tokens = try session.tokenize(text: "hello", addSpecial: true)

        XCTAssertEqual(tokens, [10, 20, 30, 40, 50])
        XCTAssertEqual(tokens.count, 5)
        XCTAssertEqual(engine.lastTokenizeText, "hello")
    }

    func testL2T2RoundTripTokenizeDetokenize() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [10, 20, 30]
        engine.detokenizeResult = "hello world"
        let session = makeSession(engine: engine)

        let tokens = try session.tokenize(text: "hello world", addSpecial: true)
        let text = try session.detokenize(tokens: tokens)

        XCTAssertEqual(tokens, [10, 20, 30])
        XCTAssertEqual(text, "hello world")
    }

    func testL2T3GenerateReturnsNonEmptyString() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 10, 11]
        engine.generateResult = ([5, 6, 7], .maxTokens)
        engine.detokenizeResult = "output text"
        let session = makeSession(engine: engine)

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 3,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "output text")
        XCTAssertEqual(result.tokenCount, 3)
        XCTAssertEqual(result.stopReason, .maxTokens)
    }

    func testL2T4TemperatureZeroForwardsGreedySamplerSetting() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 10, 11]
        engine.generateResult = ([5, 6, 7], .maxTokens)
        engine.detokenizeResult = "output text"
        let session = makeSession(engine: engine)

        _ = try session.generate(
            prompt: "prompt",
            maxTokens: 3,
            stopSequences: [],
            sampler: SamplerConfig(temperature: 0)
        )

        XCTAssertEqual(engine.lastGenerateSampler?.temperature, 0)
    }

    func testL2T5GenerateForwardsMaxTokens() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 10, 11]
        engine.generateResult = ([5, 6, 7, 8, 9], .maxTokens)
        engine.detokenizeResult = "five tokens"
        let session = makeSession(engine: engine)

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 5,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.tokenCount, 5)
        XCTAssertEqual(result.stopReason, .maxTokens)
        XCTAssertEqual(engine.lastGenerateMaxTokens, 5)
    }

    func testL2T6StopSequenceTruncatesText() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 10, 11]
        engine.generateResult = ([5, 6, 7], .maxTokens)
        engine.detokenizeResult = "hello STOP world"
        let session = makeSession(engine: engine)

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 3,
            stopSequences: ["STOP"],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "hello ")
        XCTAssertEqual(result.stopReason, .stopSequence)
    }

    func testL2T7EosStopReasonPassesThrough() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 10, 11]
        engine.generateResult = ([5, 6], .eos)
        engine.detokenizeResult = "done"
        let session = makeSession(engine: engine)

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 2,
            stopSequences: ["STOP"],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "done")
        XCTAssertEqual(result.stopReason, .eos)
    }

    func testL2T8PromptOverflowThrowsContextOverflow() {
        let engine = MockLlamaEngine()
        engine.nCtxValue = 4
        engine.tokenizeResult = [1, 2, 3, 4, 5]
        let session = makeSession(engine: engine)

        XCTAssertThrowsError(
            try session.generate(
                prompt: "prompt",
                maxTokens: 1,
                stopSequences: [],
                sampler: SamplerConfig()
            )
        ) { error in
            guard case .contextOverflow(let promptTokens, let contextSize) = error as? LlamaError else {
                return XCTFail("Expected contextOverflow, got \(error)")
            }

            XCTAssertEqual(promptTokens, 5)
            XCTAssertEqual(contextSize, 4)
        }
    }

    private func makeSession(engine: MockLlamaEngine) -> LlamaSession {
        LlamaSession(
            sessionId: "test-session",
            engine: engine,
            metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: false),
            priority: .interactive
        )
    }
}

private final class MockLlamaEngine: LlamaEngine, @unchecked Sendable {
    var nCtxValue: UInt32 = 64
    var tokenizeResult: [Int32] = []
    var detokenizeResult = ""
    var generateResult: (tokens: [Int32], stopReason: StopReason) = ([], .maxTokens)
    var eosTokenValue: Int32 = 2

    var lastTokenizeText: String?
    var lastGeneratePromptTokens: [Int32]?
    var lastGenerateMaxTokens: Int?
    var lastGenerateSampler: SamplerConfig?

    var nCtx: UInt32 {
        nCtxValue
    }

    func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        _ = addSpecial
        lastTokenizeText = text
        return tokenizeResult
    }

    func detokenize(tokens: [Int32]) throws -> String {
        _ = tokens
        return detokenizeResult
    }

    func vocabEosToken() -> Int32 {
        eosTokenValue
    }

    func isEog(token: Int32) -> Bool {
        token == eosTokenValue
    }

    func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        lastGeneratePromptTokens = promptTokens
        lastGenerateMaxTokens = maxTokens
        lastGenerateSampler = sampler
        return generateResult
    }

    func generateStreaming(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        lastGeneratePromptTokens = promptTokens
        lastGenerateMaxTokens = maxTokens
        lastGenerateSampler = sampler

        for token in generateResult.tokens.prefix(maxTokens) {
            if isCancelled() {
                return .cancelled
            }
            onToken(token)
        }

        return generateResult.stopReason
    }
}
