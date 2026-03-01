import XCTest
@testable import DustLlm
import DustCore

final class LLMStreamingTests: XCTestCase {
    func testL3T1StreamGenerateEmitsIncrementingTokenIndexes() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 2, 3]
        engine.streamingTokens = [10, 11, 12, 13, 14]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var tokenIndexes: [Int] = []

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 5,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { tokenIndex, _, _ in
                tokenIndexes.append(tokenIndex)
            },
            onComplete: { _, _, _, _ in },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertEqual(tokenIndexes, [0, 1, 2, 3, 4])
    }

    func testL3T2InferenceCompleteReportsCompletionTokensAfterLastToken() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 2]
        engine.streamingTokens = [10, 11, 12, 13, 14]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var tokenEvents = 0
        var completionCount = 0
        var completionTokens = -1

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 5,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in
                tokenEvents += 1
            },
            onComplete: { _, tokenCount, _, _ in
                completionCount += 1
                completionTokens = tokenCount
                XCTAssertEqual(tokenEvents, 5)
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertEqual(completionCount, 1)
        XCTAssertEqual(completionTokens, 5)
    }

    func testL3T3InferenceCompleteReportsPositiveTokensPerSecond() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [10, 11, 12]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var reportedTokensPerSecond = 0.0

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 3,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in },
            onComplete: { _, _, tokensPerSecond, _ in
                reportedTokensPerSecond = tokensPerSecond
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertGreaterThan(reportedTokensPerSecond, 0)
    }

    func testL3T4CancelGenerationMidStreamStopsWithCancelledReason() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [10, 11, 12, 13, 14]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var tokenEvents = 0
        var completionStopReason: StopReason?

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 5,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { tokenIndex, _, _ in
                tokenEvents += 1
                if tokenIndex == 2 {
                    session.cancelGeneration()
                }
            },
            onComplete: { _, _, _, stopReason in
                completionStopReason = stopReason
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertEqual(completionStopReason, .cancelled)
        XCTAssertLessThanOrEqual(tokenEvents, 4)
    }

    func testL3T5CancelGenerationWhileIdleIsNoOpAndSessionRemainsUsable() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1, 2]
        engine.generateResult = ([10, 11], .maxTokens)
        engine.detokenizeResult = "ok"
        let session = makeSession(engine: engine)

        session.cancelGeneration()

        XCTAssertEqual(session.status(), .ready)

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "ok")
        XCTAssertEqual(result.tokenCount, 2)
    }

    func testL3T6SecondStreamGenerateWhileBusyReportsModelNotReady() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [10, 11]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        let firstTokenReady = DispatchSemaphore(value: 0)
        let releaseFirstStream = DispatchSemaphore(value: 0)
        let firstStreamFinished = DispatchSemaphore(value: 0)

        DispatchQueue.global().async {
            session.streamGenerate(
                prompt: "prompt",
                maxTokens: 2,
                stopSequences: [],
                sampler: SamplerConfig(),
                onToken: { tokenIndex, _, _ in
                    if tokenIndex == 0 {
                        firstTokenReady.signal()
                        _ = releaseFirstStream.wait(timeout: .now() + 1)
                    }
                },
                onComplete: { _, _, _, _ in
                    firstStreamFinished.signal()
                },
                onError: { _, _ in
                    firstStreamFinished.signal()
                }
            )
        }

        XCTAssertEqual(firstTokenReady.wait(timeout: .now() + 1), .success)

        var concurrentError: Error?
        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 1,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in
                XCTFail("Second stream should not emit tokens")
            },
            onComplete: { _, _, _, _ in
                XCTFail("Second stream should not complete successfully")
            },
            onError: { error, _ in
                concurrentError = error
            }
        )

        releaseFirstStream.signal()
        XCTAssertEqual(firstStreamFinished.wait(timeout: .now() + 1), .success)

        guard let mlCoreError = concurrentError as? DustCoreError,
              case .modelNotReady = mlCoreError else {
            return XCTFail("Expected modelNotReady, got \(String(describing: concurrentError))")
        }
    }

    func testL3T7MidStreamErrorReportsFailureAndSessionRemainsUsable() throws {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [10, 11, 12, 13, 14]
        engine.streamingError = LlamaError.decodeFailed
        engine.streamingErrorAfterTokens = 3
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var tokenEvents = 0
        var failureTokenCount = -1
        var reportedError: Error?

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 5,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in
                tokenEvents += 1
            },
            onComplete: { _, _, _, _ in
                XCTFail("Expected streaming failure")
            },
            onError: { error, tokenCount in
                reportedError = error
                failureTokenCount = tokenCount
            }
        )

        XCTAssertEqual(tokenEvents, 3)
        XCTAssertEqual(failureTokenCount, 3)
        XCTAssertNotNil(reportedError)

        engine.streamingError = nil
        engine.generateResult = ([20, 21], .maxTokens)
        engine.detokenizeHandler = nil
        engine.detokenizeResult = "recovered"

        let result = try session.generate(
            prompt: "prompt",
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.text, "recovered")
        XCTAssertEqual(result.tokenCount, 2)
    }

    func testL3T8SecondStreamAfterCancelSucceeds() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [10, 11, 12]
        engine.detokenizeHandler = { tokens in
            String(repeating: "x", count: tokens.count)
        }
        let session = makeSession(engine: engine)

        var firstStopReason: StopReason?
        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 3,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { tokenIndex, _, _ in
                if tokenIndex == 0 {
                    session.cancelGeneration()
                }
            },
            onComplete: { _, _, _, stopReason in
                firstStopReason = stopReason
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        engine.streamingTokens = [20, 21]
        engine.streamingStopReason = .maxTokens
        engine.detokenizeHandler = { tokens in
            String(repeating: "y", count: tokens.count)
        }

        var secondStopReason: StopReason?
        var secondTokenCount = -1
        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 2,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, _ in },
            onComplete: { _, tokenCount, _, stopReason in
                secondStopReason = stopReason
                secondTokenCount = tokenCount
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertEqual(firstStopReason, .cancelled)
        XCTAssertEqual(secondStopReason, .maxTokens)
        XCTAssertEqual(secondTokenCount, 2)
    }

    func testL3T9MultiByteEmojiAssemblesWithoutReplacementCharacters() {
        let engine = MockLlamaEngine()
        engine.tokenizeResult = [1]
        engine.streamingTokens = [100, 101, 102, 103]
        engine.detokenizeHandler = { tokens in
            if tokens == [100] {
                return "Hello "
            }
            if tokens == [100, 101] {
                return "Hello "
            }
            if tokens == [100, 101, 102] {
                return "Hello 😀"
            }
            if tokens == [100, 101, 102, 103] {
                return "Hello 😀!"
            }
            return ""
        }
        let session = makeSession(engine: engine)

        var tokenTexts: [String] = []
        var completedText = ""

        session.streamGenerate(
            prompt: "prompt",
            maxTokens: 4,
            stopSequences: [],
            sampler: SamplerConfig(),
            onToken: { _, _, text in
                tokenTexts.append(text)
            },
            onComplete: { fullText, _, _, _ in
                completedText = fullText
            },
            onError: { error, _ in
                XCTFail("Unexpected error: \(error)")
            }
        )

        XCTAssertEqual(tokenTexts, ["Hello ", "", "😀", "!"])
        XCTAssertEqual(tokenTexts.joined(), "Hello 😀!")
        XCTAssertEqual(completedText, "Hello 😀!")
        XCTAssertFalse(completedText.contains("\u{FFFD}"))
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

    var streamingTokens: [Int32] = []
    var streamingStopReason: StopReason = .maxTokens
    var streamingError: Error?
    var streamingErrorAfterTokens = 0
    var detokenizeHandler: (([Int32]) -> String)?

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
        if let detokenizeHandler {
            return detokenizeHandler(tokens)
        }
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

        var emittedTokenCount = 0

        for token in streamingTokens.prefix(maxTokens) {
            if isCancelled() {
                return .cancelled
            }

            if let streamingError, emittedTokenCount >= streamingErrorAfterTokens {
                throw streamingError
            }

            onToken(token)
            emittedTokenCount += 1
        }

        return streamingStopReason
    }
}
