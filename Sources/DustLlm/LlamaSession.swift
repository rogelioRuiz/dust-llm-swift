import Foundation
import DustCore

public final class LlamaSession: NSObject, DustModelSession, @unchecked Sendable {
    public let sessionId: String
    public let metadata: LLMModelMetadata

    private let sessionPriority: DustSessionPriority
    private let lock = NSLock()
    private var engine: LlamaEngine?
    private(set) var visionEncoder: VisionEncoderProtocol?
    private var evicted = false
    private var currentStatus: DustModelStatus = .ready
    private var isGenerating = false
    private var cancelRequested = false
    private var storedChatMessages: [ChatMessage] = []
    private var storedContextUsed = 0
    private let templateEngine: ChatTemplateEngine

    public var contextUsed: Int {
        lock.lock()
        defer { lock.unlock() }
        return storedContextUsed
    }

    public var isModelEvicted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return evicted
    }

    public init(
        sessionId: String,
        context: LlamaContext,
        priority: DustSessionPriority,
        visionEncoder: VisionEncoderProtocol? = nil
    ) {
        self.sessionId = sessionId
        self.engine = context
        self.visionEncoder = visionEncoder
        self.metadata = context.metadata
        self.sessionPriority = priority
        self.templateEngine = ChatTemplateEngine(templateString: context.metadata.chatTemplate)
    }

    public init(
        sessionId: String,
        engine: LlamaEngine,
        metadata: LLMModelMetadata,
        priority: DustSessionPriority,
        visionEncoder: VisionEncoderProtocol? = nil
    ) {
        self.sessionId = sessionId
        self.engine = engine
        self.visionEncoder = visionEncoder
        self.metadata = metadata
        self.sessionPriority = priority
        self.templateEngine = ChatTemplateEngine(templateString: metadata.chatTemplate)
    }

    public init(
        sessionId: String,
        metadata: LLMModelMetadata,
        priority: DustSessionPriority
    ) {
        self.sessionId = sessionId
        self.engine = nil
        self.visionEncoder = nil
        self.metadata = metadata
        self.sessionPriority = priority
        self.templateEngine = ChatTemplateEngine(templateString: metadata.chatTemplate)
    }

    public func predict(inputs: [DustInputTensor]) async throws -> [DustOutputTensor] {
        _ = inputs
        throw DustCoreError.inferenceFailed(detail: "predict is not implemented in L1")
    }

    public func status() -> DustModelStatus {
        lock.lock()
        defer { lock.unlock() }
        return currentStatus
    }

    public func priority() -> DustSessionPriority {
        sessionPriority
    }

    public func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        let engine = try activeEngine()
        return try engine.tokenize(text: text, addSpecial: addSpecial)
    }

    public func detokenize(tokens: [Int32]) throws -> String {
        let engine = try activeEngine()
        return try engine.detokenize(tokens: tokens)
    }

    public func countTokens(text: String) throws -> Int {
        try tokenize(text: text, addSpecial: true).count
    }

    public func getEmbedding(text: String) throws -> [Float] {
        let engine = try activeEngine()
        let tokens = try engine.tokenize(text: text, addSpecial: true)
        return try engine.getEmbedding(tokens: tokens)
    }

    public var embeddingDims: Int {
        (try? activeEngine().embeddingDims) ?? 0
    }

    public func generate(
        prompt: String,
        imageData: Data? = nil,
        messages: [[String: String]]? = nil,
        maxTokens: Int,
        stopSequences: [String],
        sampler: SamplerConfig
    ) throws -> GenerateResult {
        let engine = try activeEngine()
        let promptTokens = try engine.tokenize(text: prompt, addSpecial: true)
        let contextSize = Int(engine.nCtx)

        guard promptTokens.count < contextSize else {
            throw LlamaError.contextOverflow(promptTokens: promptTokens.count, contextSize: contextSize)
        }

        let visionEncoder = self.visionEncoder
        if imageData != nil, !engine.supportsNativeImage, visionEncoder == nil {
            throw LlamaError.unsupportedOperation("vision input requires a vision-capable model")
        }

        try beginGeneration()
        defer { endGeneration() }

        let generated: (tokens: [Int32], stopReason: StopReason)
        if let imageData, engine.supportsNativeImage, let messages {
            generated = try engine.generateWithNativeImage(
                messages: messages,
                imageData: imageData,
                maxTokens: maxTokens,
                sampler: sampler
            )
        } else if let imageData, let visionEncoder {
            let imageEmbedding = try visionEncoder.encode(imageBytes: imageData)
            defer { visionEncoder.freeEmbedding(imageEmbedding) }

            let totalPromptTokens = promptTokens.count + imageEmbedding.tokenCount
            guard totalPromptTokens < contextSize else {
                throw LlamaError.contextOverflow(promptTokens: totalPromptTokens, contextSize: contextSize)
            }

            generated = try engine.generateWithVision(
                promptTokens: promptTokens,
                imageEmbedding: imageEmbedding,
                visionEncoder: visionEncoder,
                maxTokens: maxTokens,
                sampler: sampler
            )
        } else {
            generated = try engine.generate(
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                sampler: sampler
            )
        }

        var text = try engine.detokenize(tokens: generated.tokens)
        var stopReason = generated.stopReason

        if !stopSequences.isEmpty, stopReason != .eos {
            for sequence in stopSequences {
                if let range = text.range(of: sequence) {
                    text = String(text[..<range.lowerBound])
                    stopReason = .stopSequence
                    break
                }
            }
        }

        return GenerateResult(
            text: text,
            tokenCount: generated.tokens.count,
            stopReason: stopReason
        )
    }

    public func applyTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool,
        enableThinking: Bool? = nil
    ) throws -> (prompt: String, tokenCount: Int) {
        let prompt = try templateEngine.apply(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            enableThinking: enableThinking
        )
        let tokens = try tokenize(text: prompt, addSpecial: true)
        return (prompt, tokens.count)
    }

    public func generateChat(
        messages: [ChatMessage],
        maxTokens: Int,
        stopSequences: [String],
        sampler: SamplerConfig,
        enableThinking: Bool? = nil
    ) throws -> (result: GenerateResult, contextUsed: Int) {
        let engine = try activeEngine()
        let history = snapshotChatMessages()
        var allMessages = history + messages
        allMessages = try trimHistory(
            messages: allMessages,
            maxTokens: maxTokens,
            contextSize: Int(engine.nCtx)
        )

        let prompt = try templateEngine.apply(
            messages: allMessages,
            addGenerationPrompt: true,
            enableThinking: enableThinking
        )
        let result = try generate(
            prompt: prompt,
            maxTokens: maxTokens,
            stopSequences: stopSequences,
            sampler: sampler
        )

        let updatedHistory = allMessages + [ChatMessage(role: "assistant", content: result.text)]
        let fullPrompt = try templateEngine.apply(
            messages: updatedHistory,
            addGenerationPrompt: false,
            enableThinking: enableThinking
        )
        let updatedContextUsed = try countTokens(text: fullPrompt)

        lock.lock()
        storedChatMessages = updatedHistory
        storedContextUsed = updatedContextUsed
        lock.unlock()

        return (result, updatedContextUsed)
    }

    public func streamGenerate(
        prompt: String,
        maxTokens: Int,
        stopSequences: [String],
        sampler: SamplerConfig,
        onToken: @escaping (_ tokenIndex: Int, _ tokenId: Int32, _ text: String) -> Void,
        onComplete: @escaping (_ fullText: String, _ tokenCount: Int, _ tokensPerSecond: Double, _ stopReason: StopReason) -> Void,
        onError: @escaping (_ error: Error, _ tokenCount: Int) -> Void
    ) {
        streamGenerate(
            prompt: prompt,
            imageData: nil,
            maxTokens: maxTokens,
            stopSequences: stopSequences,
            sampler: sampler,
            onToken: onToken,
            onComplete: { fullText, tokenCount, _, tokensPerSecond, stopReason in
                onComplete(fullText, tokenCount, tokensPerSecond, stopReason)
            },
            onError: onError
        )
    }

    public func streamGenerate(
        prompt: String,
        imageData: Data? = nil,
        messages: [[String: String]]? = nil,
        maxTokens: Int,
        stopSequences: [String],
        sampler: SamplerConfig,
        onToken: @escaping (_ tokenIndex: Int, _ tokenId: Int32, _ text: String) -> Void,
        onComplete: @escaping (_ fullText: String, _ tokenCount: Int, _ promptTokenCount: Int, _ tokensPerSecond: Double, _ stopReason: StopReason) -> Void,
        onError: @escaping (_ error: Error, _ tokenCount: Int) -> Void
    ) {
        let engine: LlamaEngine
        do {
            engine = try activeEngine()
        } catch {
            onError(error, 0)
            return
        }

        let promptTokens: [Int32]
        do {
            promptTokens = try engine.tokenize(text: prompt, addSpecial: true)
        } catch {
            onError(error, 0)
            return
        }

        let contextSize = Int(engine.nCtx)
        guard promptTokens.count < contextSize else {
            onError(LlamaError.contextOverflow(promptTokens: promptTokens.count, contextSize: contextSize), 0)
            return
        }

        let visionEncoder = self.visionEncoder
        if imageData != nil, !engine.supportsNativeImage, visionEncoder == nil {
            onError(LlamaError.unsupportedOperation("vision input requires a vision-capable model"), 0)
            return
        }

        do {
            try beginGeneration()
        } catch {
            onError(error, 0)
            return
        }

        defer { endGeneration() }

        let startTime = CFAbsoluteTimeGetCurrent()
        var generatedTokens: [Int32] = []
        var lastDetokenizedText = ""
        var emittedText = ""
        var completionText = ""
        var stoppedBySequence = false
        var streamingFailure: Error?
        var promptTokenCount = promptTokens.count

        // Shared token handler used by all generation paths.
        let handleToken: (Int32) -> Void = { token in
            if streamingFailure != nil {
                self.requestCancellation()
                return
            }

            generatedTokens.append(token)

            let detokenizedText: String
            do {
                detokenizedText = try engine.detokenize(tokens: generatedTokens)
            } catch {
                streamingFailure = error
                self.requestCancellation()
                return
            }

            lastDetokenizedText = detokenizedText
            var nextCompletionText = detokenizedText

            if !stopSequences.isEmpty {
                for sequence in stopSequences {
                    if let range = detokenizedText.range(of: sequence) {
                        nextCompletionText = String(detokenizedText[..<range.lowerBound])
                        stoppedBySequence = true
                        self.requestCancellation()
                        break
                    }
                }
            }

            let tokenText: String
            if nextCompletionText.hasPrefix(emittedText) {
                tokenText = String(nextCompletionText.dropFirst(emittedText.count))
                emittedText = nextCompletionText
            } else if !stoppedBySequence {
                tokenText = detokenizedText.hasPrefix(emittedText)
                    ? String(detokenizedText.dropFirst(emittedText.count))
                    : ""
                emittedText = detokenizedText
            } else {
                tokenText = ""
            }

            completionText = nextCompletionText
            onToken(generatedTokens.count - 1, token, tokenText)
        }

        do {
            let rawStopReason: StopReason
            if let imageData, engine.supportsNativeImage, let messages {
                rawStopReason = try engine.generateStreamingWithNativeImage(
                    messages: messages,
                    imageData: imageData,
                    maxTokens: maxTokens,
                    sampler: sampler,
                    isCancelled: {
                        self.isCancellationRequested()
                    },
                    onToken: handleToken
                )
            } else if let imageData, let visionEncoder {
                let imageEmbedding = try visionEncoder.encode(imageBytes: imageData)
                defer { visionEncoder.freeEmbedding(imageEmbedding) }

                promptTokenCount += imageEmbedding.tokenCount
                if promptTokenCount >= contextSize {
                    throw LlamaError.contextOverflow(promptTokens: promptTokenCount, contextSize: contextSize)
                }

                rawStopReason = try engine.generateStreamingWithVision(
                    promptTokens: promptTokens,
                    imageEmbedding: imageEmbedding,
                    visionEncoder: visionEncoder,
                    maxTokens: maxTokens,
                    sampler: sampler,
                    isCancelled: {
                        self.isCancellationRequested()
                    },
                    onToken: handleToken
                )
            } else {
                rawStopReason = try engine.generateStreaming(
                    promptTokens: promptTokens,
                    maxTokens: maxTokens,
                    sampler: sampler,
                    isCancelled: {
                        self.isCancellationRequested()
                    },
                    onToken: handleToken
                )
            }

            if let streamingFailure {
                onError(streamingFailure, generatedTokens.count)
                return
            }

            let elapsedSeconds = max(CFAbsoluteTimeGetCurrent() - startTime, Double.leastNonzeroMagnitude)
            let tokenCount = generatedTokens.count
            let stopReason = stoppedBySequence ? .stopSequence : rawStopReason
            let finalText = stoppedBySequence ? completionText : lastDetokenizedText
            let tokensPerSecond = tokenCount == 0 ? 0 : Double(tokenCount) / elapsedSeconds

            onComplete(finalText, tokenCount, promptTokenCount, tokensPerSecond, stopReason)
        } catch {
            onError(streamingFailure ?? error, generatedTokens.count)
        }
    }

    public func cancelGeneration() {
        requestCancellation()
    }

    public func clearHistory() {
        lock.lock()
        storedChatMessages = []
        storedContextUsed = 0
        lock.unlock()
    }

    public func close() async throws {
        let retainedVisionEncoder: VisionEncoderProtocol?

        lock.lock()
        currentStatus = .unloading
        isGenerating = false
        cancelRequested = false
        storedChatMessages = []
        storedContextUsed = 0
        retainedVisionEncoder = visionEncoder
        engine = nil
        visionEncoder = nil
        lock.unlock()

        retainedVisionEncoder?.close()
    }

    public func evict() {
        let retainedVisionEncoder: VisionEncoderProtocol?

        lock.lock()
        evicted = true
        isGenerating = false
        cancelRequested = false
        storedChatMessages = []
        storedContextUsed = 0
        retainedVisionEncoder = visionEncoder
        engine = nil
        visionEncoder = nil
        lock.unlock()

        retainedVisionEncoder?.close()
    }

    private func trimHistory(
        messages: [ChatMessage],
        maxTokens: Int,
        contextSize: Int
    ) throws -> [ChatMessage] {
        let prompt = try templateEngine.apply(messages: messages, addGenerationPrompt: true)
        let promptTokens = try countTokens(text: prompt)

        if promptTokens + maxTokens <= contextSize {
            return messages
        }

        let nonSystemIndexes = messages.enumerated()
            .compactMap { entry in
                entry.element.role == "system" ? nil : entry.offset
            }

        guard nonSystemIndexes.count > 1 else {
            throw LlamaError.contextOverflow(promptTokens: promptTokens, contextSize: contextSize)
        }

        var trimmedCount = 2
        while trimmedCount <= nonSystemIndexes.count {
            let removedIndexes = Set(nonSystemIndexes.prefix(trimmedCount))
            let candidate = messages.enumerated()
                .filter { !removedIndexes.contains($0.offset) }
                .map(\.element)
            let candidatePrompt = try templateEngine.apply(messages: candidate, addGenerationPrompt: true)
            let candidateTokens = try countTokens(text: candidatePrompt)

            if candidateTokens + maxTokens <= contextSize {
                return candidate
            }

            trimmedCount += 2
        }

        guard let lastNonSystemIndex = nonSystemIndexes.last else {
            throw LlamaError.contextOverflow(promptTokens: promptTokens, contextSize: contextSize)
        }

        let candidate = messages.enumerated()
            .filter { entry in
                entry.element.role == "system" || entry.offset == lastNonSystemIndex
            }
            .map(\.element)
        let candidatePrompt = try templateEngine.apply(messages: candidate, addGenerationPrompt: true)
        let candidateTokens = try countTokens(text: candidatePrompt)

        if candidateTokens + maxTokens <= contextSize {
            return candidate
        }

        throw LlamaError.contextOverflow(promptTokens: candidateTokens, contextSize: contextSize)
    }

    private func snapshotChatMessages() -> [ChatMessage] {
        lock.lock()
        defer { lock.unlock() }
        return storedChatMessages
    }

    private func beginGeneration() throws {
        lock.lock()
        defer { lock.unlock() }

        guard !evicted else {
            throw LlamaError.modelEvicted
        }

        guard currentStatus == .ready, !isGenerating else {
            throw DustCoreError.modelNotReady
        }

        isGenerating = true
        cancelRequested = false
    }

    private func endGeneration() {
        lock.lock()
        isGenerating = false
        cancelRequested = false
        lock.unlock()
    }

    private func requestCancellation() {
        lock.lock()
        if isGenerating {
            cancelRequested = true
        }
        lock.unlock()
    }

    private func isCancellationRequested() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelRequested
    }

    private func activeEngine() throws -> LlamaEngine {
        lock.lock()
        defer { lock.unlock() }

        guard !evicted else {
            throw LlamaError.modelEvicted
        }

        guard let engine else {
            throw DustCoreError.sessionClosed
        }

        return engine
    }
}
