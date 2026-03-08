#if canImport(MLXLLM)
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
#if canImport(MLXVLM)
import MLXVLM
#endif

@available(iOS 17.0, macOS 14.0, *)
public final class MLXEngine: @unchecked Sendable {
    private let container: ModelContainer
    private let modelPath: String
    private let contextSize: UInt32
    private let eosId: Int32
    private let isVLM: Bool
    public let metadata: LLMModelMetadata
    /// Cached tokenizer used to avoid re-entering the ModelContainer lock during generation.
    private var cachedTokenizer: (any Tokenizer)?

    public init(path: String, config: LLMConfig) throws {
        self.modelPath = path

        #if targetEnvironment(simulator)
        throw LlamaError.unsupportedOperation(
            "MLX requires Metal GPU which is not available in the iOS Simulator. Use a physical device."
        )
        #endif

        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: path)
        )

        var isVLM = false
        let hasVisionConfig = MLXModelDetector.isVLMModel(from: path)

        var loadedContainer: ModelContainer?
        var loadError: Error?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                #if canImport(MLXVLM)
                if hasVisionConfig {
                    // Models with vision_config (e.g. Qwen3.5 early-fusion VLMs)
                    // must be loaded via VLMModelFactory so that ctx.processor can
                    // handle image inputs.  Fall back to LLM if VLM load fails.
                    do {
                        loadedContainer = try await VLMModelFactory.shared.loadContainer(
                            configuration: configuration
                        )
                        isVLM = true
                    } catch {
                        // VLM load failed — try LLM as fallback for text-only use.
                        loadedContainer = try await LLMModelFactory.shared.loadContainer(
                            configuration: configuration
                        )
                    }
                } else {
                    loadedContainer = try await LLMModelFactory.shared.loadContainer(
                        configuration: configuration
                    )
                }
                #else
                loadedContainer = try await LLMModelFactory.shared.loadContainer(
                    configuration: configuration
                )
                #endif
            } catch {
                loadError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let loadError {
            throw loadError
        }
        guard let container = loadedContainer else {
            throw LlamaError.loadFailed(path: path)
        }
        self.container = container
        self.isVLM = isVLM

        // Read EOS token and cache tokenizer from container
        var resolvedEos: Int32 = -1
        var resolvedTokenizer: (any Tokenizer)?
        let eosSema = DispatchSemaphore(value: 0)
        Task {
            (resolvedEos, resolvedTokenizer) = await container.perform { ctx in
                (Int32(ctx.tokenizer.eosTokenId ?? -1), ctx.tokenizer)
            }
            eosSema.signal()
        }
        eosSema.wait()
        self.eosId = resolvedEos
        self.cachedTokenizer = resolvedTokenizer

        // Read metadata from config files on disk
        let chatTemplate = MLXModelDetector.readChatTemplate(from: path)
        let name = MLXModelDetector.readModelName(from: path)
        self.metadata = LLMModelMetadata(
            name: name,
            chatTemplate: chatTemplate,
            hasVision: isVLM
        )

        let maxPos = MLXModelDetector.readMaxPositionEmbeddings(from: path)
        self.contextSize = UInt32(maxPos ?? Int(config.contextSize))
    }

    private func syncPerform<R>(_ action: @Sendable @escaping (ModelContext) throws -> R) throws -> R {
        var result: R?
        var caught: Error?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                result = try await container.perform { ctx in
                    try action(ctx)
                }
            } catch {
                caught = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let caught { throw caught }
        return result!
    }

    private func syncPerformAsync<R>(_ action: @Sendable @escaping (ModelContext) async throws -> R) throws -> R {
        var result: R?
        var caught: Error?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                result = try await container.perform { ctx in
                    try await action(ctx)
                }
            } catch {
                caught = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let caught { throw caught }
        return result!
    }

    private static func mapParameters(_ sampler: SamplerConfig, maxTokens: Int) -> GenerateParameters {
        var params = GenerateParameters()
        params.temperature = sampler.temperature
        params.topP = sampler.topP
        if sampler.repeatPenalty != 1.0 {
            params.repetitionPenalty = sampler.repeatPenalty
            params.repetitionContextSize = Int(sampler.repeatLastN)
        }
        // topK and minP are not exposed in GenerateParameters
        return params
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension MLXEngine: LlamaEngine {
    public var nCtx: UInt32 { contextSize }

    public func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        guard let tokenizer = cachedTokenizer else {
            throw LlamaError.loadFailed(path: modelPath)
        }
        let tokens: [Int] = tokenizer.encode(text: text, addSpecialTokens: addSpecial)
        return tokens.map { Int32($0) }
    }

    public func detokenize(tokens: [Int32]) throws -> String {
        guard let tokenizer = cachedTokenizer else {
            throw LlamaError.loadFailed(path: modelPath)
        }
        return tokenizer.decode(tokens: tokens.map { Int($0) })
    }

    public func vocabEosToken() -> Int32 { eosId }

    public func isEog(token: Int32) -> Bool { token == eosId }

    public func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        let params = Self.mapParameters(sampler, maxTokens: maxTokens)

        let result: (tokens: [Int32], stopReason: StopReason) = try syncPerformAsync { ctx in
            let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
            var generatedTokens: [Int32] = []
            var reason: StopReason = .maxTokens

            _ = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: ctx
            ) { tokens in
                guard let last = tokens.last else { return .more }
                generatedTokens.append(Int32(last))

                if let eosId = ctx.tokenizer.eosTokenId, last == eosId {
                    reason = .eos
                    return .stop
                }
                return generatedTokens.count >= maxTokens ? .stop : .more
            }

            return (generatedTokens, reason)
        }

        return result
    }

    public func generateStreaming(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        let params = Self.mapParameters(sampler, maxTokens: maxTokens)

        // withoutActuallyEscaping is safe here because syncPerformAsync blocks
        // until the closure completes (semaphore-based synchronous bridge).
        return try withoutActuallyEscaping(isCancelled) { escapableIsCancelled in
            try withoutActuallyEscaping(onToken) { escapableOnToken in
                let reason: StopReason = try syncPerformAsync { ctx in
                    let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
                    var count = 0
                    var stopReason: StopReason = .maxTokens

                    _ = try MLXLMCommon.generate(
                        input: input,
                        parameters: params,
                        context: ctx
                    ) { tokens in
                        guard let last = tokens.last else { return .more }
                        let token = Int32(last)
                        count += 1

                        if let eosId = ctx.tokenizer.eosTokenId, last == eosId {
                            stopReason = .eos
                            return .stop
                        }

                        escapableOnToken(token)

                        if escapableIsCancelled() {
                            stopReason = .cancelled
                            return .stop
                        }

                        return count >= maxTokens ? .stop : .more
                    }

                    return stopReason
                }

                return reason
            }
        }
    }

    public var supportsNativeImage: Bool { isVLM }

    private static func buildUserInput(
        messages: [[String: String]],
        imageData: Data
    ) -> UserInput {
        let ciImage = CIImage(data: imageData)!
        // Build Chat.Message array; attach image to the last user message.
        var chatMessages: [Chat.Message] = []
        let lastUserIndex = messages.lastIndex { ($0["role"] ?? "") == "user" }
        for (i, msg) in messages.enumerated() {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            let chatRole: Chat.Message.Role
            switch role {
            case "system": chatRole = .system
            case "assistant": chatRole = .assistant
            default: chatRole = .user
            }
            let images: [UserInput.Image] = (i == lastUserIndex) ? [.ciImage(ciImage)] : []
            chatMessages.append(Chat.Message(role: chatRole, content: content, images: images))
        }
        return UserInput(chat: chatMessages)
    }

    public func generateWithNativeImage(
        messages: [[String: String]],
        imageData: Data,
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        let params = Self.mapParameters(sampler, maxTokens: maxTokens)
        let userInput = Self.buildUserInput(messages: messages, imageData: imageData)

        let result: (tokens: [Int32], stopReason: StopReason) = try syncPerformAsync { ctx in
            let input = try await ctx.processor.prepare(input: userInput)

            var generatedTokens: [Int32] = []
            var reason: StopReason = .maxTokens

            _ = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: ctx
            ) { tokens in
                guard let last = tokens.last else { return .more }
                generatedTokens.append(Int32(last))

                if let eosId = ctx.tokenizer.eosTokenId, last == eosId {
                    reason = .eos
                    return .stop
                }
                return generatedTokens.count >= maxTokens ? .stop : .more
            }

            return (generatedTokens, reason)
        }

        return result
    }

    public func generateStreamingWithNativeImage(
        messages: [[String: String]],
        imageData: Data,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Bool,
        onToken: (Int32) -> Void
    ) throws -> StopReason {
        let params = Self.mapParameters(sampler, maxTokens: maxTokens)
        let userInput = Self.buildUserInput(messages: messages, imageData: imageData)

        return try withoutActuallyEscaping(isCancelled) { escapableIsCancelled in
            try withoutActuallyEscaping(onToken) { escapableOnToken in
                let reason: StopReason = try syncPerformAsync { ctx in
                    let input = try await ctx.processor.prepare(input: userInput)

                    var count = 0
                    var stopReason: StopReason = .maxTokens

                    _ = try MLXLMCommon.generate(
                        input: input,
                        parameters: params,
                        context: ctx
                    ) { tokens in
                        guard let last = tokens.last else { return .more }
                        let token = Int32(last)
                        count += 1

                        if let eosId = ctx.tokenizer.eosTokenId, last == eosId {
                            stopReason = .eos
                            return .stop
                        }

                        escapableOnToken(token)

                        if escapableIsCancelled() {
                            stopReason = .cancelled
                            return .stop
                        }

                        return count >= maxTokens ? .stop : .more
                    }

                    return stopReason
                }

                return reason
            }
        }
    }
}
#endif
