#if canImport(MLXLLM)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@available(iOS 17.0, macOS 14.0, *)
public final class MLXEngine: @unchecked Sendable {
    private let container: ModelContainer
    private let modelPath: String
    private let contextSize: UInt32
    private let eosId: Int32
    public let metadata: LLMModelMetadata

    public init(path: String, config: LLMConfig) throws {
        self.modelPath = path

        let configuration = ModelConfiguration(
            id: path,
            directory: URL(fileURLWithPath: path)
        )

        var loadedContainer: ModelContainer?
        var loadError: Error?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                loadedContainer = try await LLMModelFactory.shared.loadContainer(
                    configuration: configuration
                )
            } catch {
                loadError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        guard let container = loadedContainer else {
            throw LlamaError.loadFailed(path: path)
        }
        self.container = container

        // Read EOS token from container
        var resolvedEos: Int32 = -1
        let eosSema = DispatchSemaphore(value: 0)
        Task {
            resolvedEos = Int32(await container.perform { ctx in
                ctx.tokenizer.eosTokenId ?? -1
            })
            eosSema.signal()
        }
        eosSema.wait()
        self.eosId = resolvedEos

        // Read metadata from config files on disk
        let chatTemplate = MLXModelDetector.readChatTemplate(from: path)
        let name = MLXModelDetector.readModelName(from: path)
        self.metadata = LLMModelMetadata(
            name: name,
            chatTemplate: chatTemplate,
            hasVision: false
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
        let tokens: [Int] = try syncPerform { ctx in
            ctx.tokenizer.encode(text: text, addSpecialTokens: addSpecial)
        }
        return tokens.map { Int32($0) }
    }

    public func detokenize(tokens: [Int32]) throws -> String {
        try syncPerform { ctx in
            ctx.tokenizer.decode(tokens: tokens.map { Int($0) })
        }
    }

    public func vocabEosToken() -> Int32 { eosId }

    public func isEog(token: Int32) -> Bool { token == eosId }

    public func generate(
        promptTokens: [Int32],
        maxTokens: Int,
        sampler: SamplerConfig
    ) throws -> (tokens: [Int32], stopReason: StopReason) {
        let params = Self.mapParameters(sampler, maxTokens: maxTokens)

        let result: (tokens: [Int32], stopReason: StopReason) = try syncPerform { ctx in
            let input = try Self.tokensToLMInput(promptTokens, context: ctx)
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

        let reason: StopReason = try syncPerform { ctx in
            let input = try Self.tokensToLMInput(promptTokens, context: ctx)
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

                onToken(token)

                if isCancelled() {
                    stopReason = .cancelled
                    return .stop
                }

                return count >= maxTokens ? .stop : .more
            }

            return stopReason
        }

        return reason
    }

    private static func tokensToLMInput(_ tokens: [Int32], context: ModelContext) throws -> LMInput {
        let mlxTokens = MLXArray(tokens.map { Int32($0) })
        return LMInput(tokens: mlxTokens)
    }
}
#endif
