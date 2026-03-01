import XCTest
@testable import DustLlm
import DustCore

final class LLMChatTemplateTests: XCTestCase {
    func testL4T1ChatMLTemplateAppliesThreeMessageConversation() throws {
        let engine = ChatTemplateEngine(templateString: ChatTemplateEngine.chatMLTemplate)
        let messages = [
            ChatMessage(role: "system", content: "You are a helpful assistant"),
            ChatMessage(role: "user", content: "Hello"),
            ChatMessage(role: "assistant", content: "Hi there"),
        ]

        let output = try engine.apply(messages: messages, addGenerationPrompt: true)

        XCTAssertEqual(
            output,
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    func testL4T2NilTemplateFallsBackToChatML() throws {
        let engine = ChatTemplateEngine(templateString: nil)

        let output = try engine.apply(
            messages: [ChatMessage(role: "user", content: "Hello")],
            addGenerationPrompt: true
        )

        XCTAssertTrue(output.contains("<|im_start|>"))
        XCTAssertTrue(output.contains("<|im_end|>"))
        XCTAssertTrue(output.contains("Hello"))
    }

    func testL4T3TrimHistoryPreservesSystemAndEvictsOldestPair() throws {
        let engine = ChatTemplateMockLlamaEngine()
        engine.nCtxValue = 200
        engine.tokenizeHandler = { text, _ in
            Array(repeating: 1, count: text.utf8.count)
        }
        let completions = [
            Int32(11): "B",
            Int32(12): "D",
            Int32(13): "F",
        ]
        engine.detokenizeHandler = { tokens in
            completions[tokens.last ?? -1] ?? ""
        }
        var nextToken: Int32 = 11
        engine.generateHandler = { _, _ in
            defer { nextToken += 1 }
            return ([nextToken], .maxTokens)
        }

        let session = makeSession(engine: engine)
        _ = try session.generateChat(
            messages: [
                ChatMessage(role: "system", content: "Hi"),
                ChatMessage(role: "user", content: "A"),
            ],
            maxTokens: 10,
            stopSequences: [],
            sampler: SamplerConfig()
        )
        _ = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "C")],
            maxTokens: 10,
            stopSequences: [],
            sampler: SamplerConfig()
        )
        _ = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "E")],
            maxTokens: 10,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        guard let trimmedPrompt = engine.generatePromptTexts.last else {
            return XCTFail("Expected a generation prompt")
        }

        XCTAssertTrue(trimmedPrompt.contains("<|im_start|>system\nHi"))
        XCTAssertFalse(trimmedPrompt.contains("<|im_start|>user\nA"))
        XCTAssertFalse(trimmedPrompt.contains("<|im_start|>assistant\nB"))
        XCTAssertTrue(trimmedPrompt.contains("<|im_start|>user\nC"))
        XCTAssertTrue(trimmedPrompt.contains("<|im_start|>assistant\nD"))
        XCTAssertTrue(trimmedPrompt.contains("<|im_start|>user\nE"))
    }

    func testL4T4SingleMessageExactFitSucceeds() throws {
        let engine = ChatTemplateMockLlamaEngine()
        engine.nCtxValue = 100
        engine.tokenizeHandler = { _, _ in
            Array(repeating: 1, count: 80)
        }
        engine.generateResult = ([21], .maxTokens)
        engine.detokenizeHandler = { _ in "ok" }
        let session = makeSession(engine: engine)

        let result = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "test")],
            maxTokens: 20,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(result.result.text, "ok")
        XCTAssertEqual(result.result.stopReason, .maxTokens)
    }

    func testL4T5SingleMessageOverflowThrowsContextOverflow() {
        let engine = ChatTemplateMockLlamaEngine()
        engine.nCtxValue = 50
        engine.tokenizeHandler = { _, _ in
            Array(repeating: 1, count: 60)
        }
        let session = makeSession(engine: engine)

        XCTAssertThrowsError(
            try session.generateChat(
                messages: [ChatMessage(role: "user", content: "huge")],
                maxTokens: 10,
                stopSequences: [],
                sampler: SamplerConfig()
            )
        ) { error in
            guard case .contextOverflow(let promptTokens, let contextSize) = error as? LlamaError else {
                return XCTFail("Expected contextOverflow, got \(error)")
            }

            XCTAssertEqual(promptTokens, 60)
            XCTAssertEqual(contextSize, 50)
        }
    }

    func testL4T6ClearHistoryResetsContextUsed() throws {
        let engine = ChatTemplateMockLlamaEngine()
        engine.tokenizeHandler = { text, _ in
            Array(repeating: 1, count: text.utf8.count)
        }
        let completions = [
            Int32(31): "Hello",
            Int32(32): "Again",
        ]
        engine.detokenizeHandler = { tokens in
            completions[tokens.last ?? -1] ?? ""
        }
        var nextToken: Int32 = 31
        engine.generateHandler = { _, _ in
            defer { nextToken += 1 }
            return ([nextToken], .maxTokens)
        }
        let session = makeSession(engine: engine)

        _ = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "Hi")],
            maxTokens: 8,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertGreaterThan(session.contextUsed, 0)

        session.clearHistory()

        XCTAssertEqual(session.contextUsed, 0)

        let secondTurn = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "Fresh")],
            maxTokens: 8,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertEqual(secondTurn.result.text, "Again")
        XCTAssertGreaterThan(session.contextUsed, 0)
    }

    func testL4T7ContextUsedIncreasesAcrossTurns() throws {
        let engine = ChatTemplateMockLlamaEngine()
        engine.nCtxValue = 512
        engine.tokenizeHandler = { text, _ in
            Array(repeating: 1, count: text.utf8.count)
        }
        let completions = [
            Int32(41): "Hi",
            Int32(42): "Fine",
        ]
        engine.detokenizeHandler = { tokens in
            completions[tokens.last ?? -1] ?? ""
        }
        var nextToken: Int32 = 41
        engine.generateHandler = { _, _ in
            defer { nextToken += 1 }
            return ([nextToken], .maxTokens)
        }
        let session = makeSession(engine: engine)

        let turnOne = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "Hello")],
            maxTokens: 8,
            stopSequences: [],
            sampler: SamplerConfig()
        )
        let turnTwo = try session.generateChat(
            messages: [ChatMessage(role: "user", content: "How are you?")],
            maxTokens: 8,
            stopSequences: [],
            sampler: SamplerConfig()
        )

        XCTAssertGreaterThan(turnOne.contextUsed, 0)
        XCTAssertGreaterThan(turnTwo.contextUsed, turnOne.contextUsed)
    }

    func testL4T8AddGenerationPromptAddsAssistantPrefix() throws {
        let engine = ChatTemplateEngine(templateString: ChatTemplateEngine.chatMLTemplate)
        let messages = [ChatMessage(role: "user", content: "Hello")]

        let withPrompt = try engine.apply(messages: messages, addGenerationPrompt: true)
        let withoutPrompt = try engine.apply(messages: messages, addGenerationPrompt: false)

        XCTAssertTrue(withPrompt.hasSuffix("<|im_start|>assistant\n"))
        XCTAssertFalse(withoutPrompt.hasSuffix("<|im_start|>assistant\n"))
    }

    private func makeSession(engine: ChatTemplateMockLlamaEngine) -> LlamaSession {
        LlamaSession(
            sessionId: "chat-template-session",
            engine: engine,
            metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: false),
            priority: .interactive
        )
    }
}

private final class ChatTemplateMockLlamaEngine: LlamaEngine, @unchecked Sendable {
    var nCtxValue: UInt32 = 256
    var tokenizeResult: [Int32] = []
    var detokenizeResult = ""
    var generateResult: (tokens: [Int32], stopReason: StopReason) = ([], .maxTokens)

    var tokenizeHandler: ((_ text: String, _ addSpecial: Bool) -> [Int32])?
    var detokenizeHandler: ((_ tokens: [Int32]) -> String)?
    var generateHandler: ((_ promptTokens: [Int32], _ maxTokens: Int) -> (tokens: [Int32], stopReason: StopReason))?

    var lastTokenizeText: String?
    var generatePromptTexts: [String] = []

    var nCtx: UInt32 {
        nCtxValue
    }

    func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        lastTokenizeText = text
        if let tokenizeHandler {
            return tokenizeHandler(text, addSpecial)
        }
        return tokenizeResult
    }

    func detokenize(tokens: [Int32]) throws -> String {
        if let detokenizeHandler {
            return detokenizeHandler(tokens)
        }
        return detokenizeResult
    }

    func vocabEosToken() -> Int32 {
        2
    }

    func isEog(token: Int32) -> Bool {
        token == 2
    }

    func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        _ = sampler
        generatePromptTexts.append(lastTokenizeText ?? "")
        if let generateHandler {
            return generateHandler(promptTokens, maxTokens)
        }
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
        _ = isCancelled
        _ = onToken
        return .maxTokens
    }
}
