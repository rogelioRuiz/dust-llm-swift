import Foundation
import llama

public protocol LlamaEngine: AnyObject, Sendable {
    var nCtx: UInt32 { get }

    func tokenize(text: String, addSpecial: Bool) throws -> [Int32]
    func detokenize(tokens: [Int32]) throws -> String
    func getEmbedding(tokens: [Int32]) throws -> [Float]
    var embeddingDims: Int { get }
    func vocabEosToken() -> Int32
    func isEog(token: Int32) -> Bool
    func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason)
    func generateStreaming(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason
    func generateWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason)
    func generateStreamingWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason
}

public struct SamplerConfig: Equatable, Sendable {
    public let temperature: Float
    public let topK: Int32
    public let topP: Float
    public let minP: Float
    public let repeatPenalty: Float
    public let repeatLastN: Int32
    public let seed: UInt32

    public init(
        temperature: Float = 0.8,
        topK: Int32 = 40,
        topP: Float = 0.95,
        minP: Float = 0.05,
        repeatPenalty: Float = 1.1,
        repeatLastN: Int32 = 64,
        seed: UInt32 = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repeatPenalty = repeatPenalty
        self.repeatLastN = repeatLastN
        self.seed = seed
    }
}

public enum StopReason: String, Equatable, Sendable {
    case maxTokens = "max_tokens"
    case stopSequence = "stop_sequence"
    case eos = "eos"
    case cancelled = "cancelled"
}

public struct GenerateResult: Equatable, Sendable {
    public let text: String
    public let tokenCount: Int
    public let stopReason: StopReason

    public init(text: String, tokenCount: Int, stopReason: StopReason) {
        self.text = text
        self.tokenCount = tokenCount
        self.stopReason = stopReason
    }
}

extension LlamaEngine {
    public func getEmbedding(tokens: [Int32]) throws -> [Float] {
        _ = tokens
        throw LlamaError.unsupportedOperation("embedding extraction is not available for this engine")
    }

    public var embeddingDims: Int {
        0
    }

    public func generateWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        _ = promptTokens
        _ = imageEmbedding
        _ = visionEncoder
        _ = maxTokens
        _ = sampler
        throw LlamaError.unsupportedOperation("vision generation is not available for this engine")
    }

    public func generateStreamingWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        _ = promptTokens
        _ = imageEmbedding
        _ = visionEncoder
        _ = maxTokens
        _ = sampler
        _ = isCancelled
        _ = onToken
        throw LlamaError.unsupportedOperation("vision generation is not available for this engine")
    }
}

extension LlamaContext: LlamaEngine {
    public var nCtx: UInt32 {
        guard let context else {
            return 0
        }

        return llama_n_ctx(context)
    }

    public func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        guard let model else {
            throw LlamaError.tokenizationFailed
        }

        let vocab = llama_model_get_vocab(model)
        let textLength = Int32(text.utf8.count)
        let requiredCount = text.withCString { pointer in
            llama_tokenize(vocab, pointer, textLength, nil, 0, addSpecial, true)
        }

        let tokenCount = requiredCount < 0 ? -requiredCount : requiredCount
        var tokens = [Int32](repeating: 0, count: Int(tokenCount))
        let actualCount = text.withCString { pointer in
            tokens.withUnsafeMutableBufferPointer { tokenBuffer in
                llama_tokenize(
                    vocab,
                    pointer,
                    textLength,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    addSpecial,
                    true
                )
            }
        }

        guard actualCount >= 0 else {
            throw LlamaError.tokenizationFailed
        }

        if actualCount == Int32(tokens.count) {
            return tokens
        }

        return Array(tokens.prefix(Int(actualCount)))
    }

    public func detokenize(tokens: [Int32]) throws -> String {
        guard let model else {
            throw LlamaError.tokenizationFailed
        }

        if tokens.isEmpty {
            return ""
        }

        let vocab = llama_model_get_vocab(model)
        var bufferLength = max(tokens.count * 8, 64)

        while true {
            var buffer = [CChar](repeating: 0, count: bufferLength)
            let actualLength = tokens.withUnsafeBufferPointer { tokenBuffer in
                buffer.withUnsafeMutableBufferPointer { textBuffer in
                    llama_detokenize(
                        vocab,
                        tokenBuffer.baseAddress,
                        Int32(tokenBuffer.count),
                        textBuffer.baseAddress,
                        Int32(textBuffer.count),
                        false,
                        true
                    )
                }
            }

            if actualLength >= 0 {
                let bytes = buffer.prefix(Int(actualLength)).map { UInt8(bitPattern: $0) }
                return String(decoding: bytes, as: UTF8.self)
            }

            let requiredLength = Int(-actualLength) + 1
            if requiredLength <= bufferLength {
                throw LlamaError.tokenizationFailed
            }
            bufferLength = requiredLength
        }
    }

    public func vocabEosToken() -> Int32 {
        guard let model else {
            return -1
        }

        return llama_vocab_eos(llama_model_get_vocab(model))
    }

    public func isEog(token: Int32) -> Bool {
        guard let model else {
            return false
        }

        return llama_vocab_is_eog(llama_model_get_vocab(model), token)
    }

    public func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        guard let model, let context else {
            throw LlamaError.decodeFailed
        }

        let contextSize = Int(llama_n_ctx(context))
        guard promptTokens.count < contextSize else {
            throw LlamaError.contextOverflow(promptTokens: promptTokens.count, contextSize: contextSize)
        }

        let vocab = llama_model_get_vocab(model)
        llama_memory_clear(llama_get_memory(context), true)

        var batch = llama_batch_init(Int32(max(promptTokens.count, 1)), 0, 1)
        defer {
            llama_batch_free(batch)
        }

        if !promptTokens.isEmpty {
            fillBatch(&batch, tokens: promptTokens, startPosition: 0)
            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        let chain = makeSampler(sampler: sampler)
        defer {
            llama_sampler_free(chain)
        }

        var generatedTokens: [Int32] = []
        var stopReason: StopReason = .maxTokens
        var nextPosition = Int32(promptTokens.count)

        for _ in 0..<maxTokens {
            let token = llama_sampler_sample(chain, context, -1)
            generatedTokens.append(token)
            llama_sampler_accept(chain, token)

            if llama_vocab_is_eog(vocab, token) {
                stopReason = .eos
                break
            }

            fillBatch(&batch, tokens: [token], startPosition: nextPosition)
            nextPosition += 1

            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        return (generatedTokens, stopReason)
    }

    public func generateStreaming(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        guard let model, let context else {
            throw LlamaError.decodeFailed
        }

        let contextSize = Int(llama_n_ctx(context))
        guard promptTokens.count < contextSize else {
            throw LlamaError.contextOverflow(promptTokens: promptTokens.count, contextSize: contextSize)
        }

        let vocab = llama_model_get_vocab(model)
        llama_memory_clear(llama_get_memory(context), true)

        var batch = llama_batch_init(Int32(max(promptTokens.count, 1)), 0, 1)
        defer {
            llama_batch_free(batch)
        }

        if !promptTokens.isEmpty {
            fillBatch(&batch, tokens: promptTokens, startPosition: 0)
            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        let chain = makeSampler(sampler: sampler)
        defer {
            llama_sampler_free(chain)
        }

        var nextPosition = Int32(promptTokens.count)

        for _ in 0..<maxTokens {
            if isCancelled() {
                return .cancelled
            }

            let token = llama_sampler_sample(chain, context, -1)
            onToken(token)
            llama_sampler_accept(chain, token)

            if llama_vocab_is_eog(vocab, token) {
                return .eos
            }

            if isCancelled() {
                return .cancelled
            }

            fillBatch(&batch, tokens: [token], startPosition: nextPosition)
            nextPosition += 1

            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        return .maxTokens
    }

    public func generateWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        guard let model, let context else {
            throw LlamaError.decodeFailed
        }

        let totalPromptTokens = promptTokens.count + imageEmbedding.tokenCount
        let contextSize = Int(llama_n_ctx(context))
        guard totalPromptTokens < contextSize else {
            throw LlamaError.contextOverflow(promptTokens: totalPromptTokens, contextSize: contextSize)
        }

        let vocab = llama_model_get_vocab(model)
        llama_memory_clear(llama_get_memory(context), true)

        var batch = llama_batch_init(Int32(max(promptTokens.count, 1)), 0, 1)
        defer {
            llama_batch_free(batch)
        }

        if !promptTokens.isEmpty {
            fillBatch(&batch, tokens: promptTokens, startPosition: 0)
            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        var nextPosition = Int32(promptTokens.count)
        try withContext { rawContext in
            try visionEncoder.evalImageEmbed(
                embedding: imageEmbedding,
                context: rawContext,
                nPast: &nextPosition
            )
        }

        let chain = makeSampler(sampler: sampler)
        defer {
            llama_sampler_free(chain)
        }

        var generatedTokens: [Int32] = []
        var stopReason: StopReason = .maxTokens

        for _ in 0..<maxTokens {
            let token = llama_sampler_sample(chain, context, -1)
            generatedTokens.append(token)
            llama_sampler_accept(chain, token)

            if llama_vocab_is_eog(vocab, token) {
                stopReason = .eos
                break
            }

            fillBatch(&batch, tokens: [token], startPosition: nextPosition)
            nextPosition += 1

            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        return (generatedTokens, stopReason)
    }

    public func generateStreamingWithVision(
        promptTokens: [Int32],
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderProtocol,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        guard let model, let context else {
            throw LlamaError.decodeFailed
        }

        let totalPromptTokens = promptTokens.count + imageEmbedding.tokenCount
        let contextSize = Int(llama_n_ctx(context))
        guard totalPromptTokens < contextSize else {
            throw LlamaError.contextOverflow(promptTokens: totalPromptTokens, contextSize: contextSize)
        }

        let vocab = llama_model_get_vocab(model)
        llama_memory_clear(llama_get_memory(context), true)

        var batch = llama_batch_init(Int32(max(promptTokens.count, 1)), 0, 1)
        defer {
            llama_batch_free(batch)
        }

        if !promptTokens.isEmpty {
            fillBatch(&batch, tokens: promptTokens, startPosition: 0)
            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        var nextPosition = Int32(promptTokens.count)
        try withContext { rawContext in
            try visionEncoder.evalImageEmbed(
                embedding: imageEmbedding,
                context: rawContext,
                nPast: &nextPosition
            )
        }

        let chain = makeSampler(sampler: sampler)
        defer {
            llama_sampler_free(chain)
        }

        for _ in 0..<maxTokens {
            if isCancelled() {
                return .cancelled
            }

            let token = llama_sampler_sample(chain, context, -1)
            onToken(token)
            llama_sampler_accept(chain, token)

            if llama_vocab_is_eog(vocab, token) {
                return .eos
            }

            if isCancelled() {
                return .cancelled
            }

            fillBatch(&batch, tokens: [token], startPosition: nextPosition)
            nextPosition += 1

            guard llama_decode(context, batch) == 0 else {
                throw LlamaError.decodeFailed
            }
        }

        return .maxTokens
    }
}

private func makeSampler(
    sampler: SamplerConfig
) -> UnsafeMutablePointer<llama_sampler> {
    let params = llama_sampler_chain_default_params()
    guard let chain = llama_sampler_chain_init(params) else {
        fatalError("llama_sampler_chain_init returned nil")
    }

    if sampler.repeatPenalty != 1.0, sampler.repeatLastN > 0 {
        if let penalties = llama_sampler_init_penalties(
            sampler.repeatLastN,
            sampler.repeatPenalty,
            0.0,
            0.0
        ) {
            llama_sampler_chain_add(chain, penalties)
        }
    }

    if sampler.temperature <= 0 {
        if let greedy = llama_sampler_init_greedy() {
            llama_sampler_chain_add(chain, greedy)
        }
        return chain
    }

    if sampler.topK > 0 {
        if let topK = llama_sampler_init_top_k(sampler.topK) {
            llama_sampler_chain_add(chain, topK)
        }
    }
    if sampler.topP < 1.0 {
        if let topP = llama_sampler_init_top_p(sampler.topP, 1) {
            llama_sampler_chain_add(chain, topP)
        }
    }
    if sampler.minP > 0.0 {
        if let minP = llama_sampler_init_min_p(sampler.minP, 1) {
            llama_sampler_chain_add(chain, minP)
        }
    }

    if let temp = llama_sampler_init_temp(sampler.temperature) {
        llama_sampler_chain_add(chain, temp)
    }
    if let dist = llama_sampler_init_dist(sampler.seed) {
        llama_sampler_chain_add(chain, dist)
    }
    return chain
}

private func fillBatch(
    _ batch: inout llama_batch,
    tokens: [Int32],
    startPosition: Int32
) {
    batch.n_tokens = 0

    for (index, token) in tokens.enumerated() {
        let batchIndex = Int(batch.n_tokens)
        batch.token[batchIndex] = token
        batch.pos[batchIndex] = startPosition + Int32(index)
        batch.n_seq_id[batchIndex] = 1
        batch.seq_id[batchIndex]![0] = 0
        batch.logits[batchIndex] = index == tokens.count - 1 ? 1 : 0
        batch.n_tokens += 1
    }
}
