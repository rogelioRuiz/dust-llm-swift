import Foundation
import llama

public struct LLMModelMetadata: Equatable, Sendable {
    public let name: String?
    public let chatTemplate: String?
    public let hasVision: Bool

    public init(name: String?, chatTemplate: String?, hasVision: Bool) {
        self.name = name
        self.chatTemplate = chatTemplate
        self.hasVision = hasVision
    }

    public func toJSObject() -> [String: Any] {
        var object: [String: Any] = ["hasVision": hasVision]
        if let name {
            object["name"] = name
        }
        if let chatTemplate {
            object["chatTemplate"] = chatTemplate
        }
        return object
    }
}

public enum LlamaError: Error, Equatable {
    case fileNotFound(path: String)
    case loadFailed(path: String)
    case contextCreationFailed(path: String)
    case contextOverflow(promptTokens: Int, contextSize: Int)
    case decodeFailed
    case tokenizationFailed
    case unsupportedOperation(String)
    case modelEvicted
}

public final class LlamaContext: @unchecked Sendable {
    private static let backendInit: Void = {
        llama_backend_init()
    }()

    public private(set) var model: OpaquePointer?
    public private(set) var context: OpaquePointer?
    public let metadata: LLMModelMetadata

    public init(path: String, config: LLMConfig) throws {
        _ = Self.backendInit

        guard FileManager.default.fileExists(atPath: path) else {
            throw LlamaError.fileNotFound(path: path)
        }

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = config.nGpuLayers

        #if targetEnvironment(simulator)
        // iOS Simulator's Metal backend has a NULL registry pointer in ggml,
        // causing SIGSEGV in ggml_backend_dev_backend_reg during load_tensors.
        // Restrict to CPU-only devices on simulator.
        var cpuDevices: [ggml_backend_dev_t?] = []
        for i in 0..<ggml_backend_dev_count() {
            if let dev = ggml_backend_dev_get(i),
               ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU {
                cpuDevices.append(dev)
            }
        }
        cpuDevices.append(nil) // NULL terminator
        #endif

        let loadedModel: OpaquePointer? = try {
            #if targetEnvironment(simulator)
            return cpuDevices.withUnsafeMutableBufferPointer { buffer in
                modelParams.devices = buffer.baseAddress
                modelParams.n_gpu_layers = 0
                return path.withCString { llama_model_load_from_file($0, modelParams) }
            }
            #else
            return path.withCString { llama_model_load_from_file($0, modelParams) }
            #endif
        }()

        guard let loadedModel else {
            throw LlamaError.loadFailed(path: path)
        }

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = config.contextSize
        contextParams.n_batch = config.batchSize
        contextParams.n_ubatch = config.batchSize

        guard let loadedContext = llama_init_from_model(loadedModel, contextParams) else {
            llama_model_free(loadedModel)
            throw LlamaError.contextCreationFailed(path: path)
        }

        model = loadedModel
        context = loadedContext
        metadata = Self.readMetadata(from: loadedModel)
    }

    deinit {
        if let context {
            llama_free(context)
        }
        if let model {
            llama_model_free(model)
        }
    }

    private static func readMetadata(from model: OpaquePointer) -> LLMModelMetadata {
        let name = readStringValue(forKey: "general.name", from: model)
        let chatTemplate = readStringValue(forKey: "tokenizer.chat_template", from: model)
        let visionValue = readStringValue(forKey: "clip.vision.image_size", from: model)

        return LLMModelMetadata(
            name: name,
            chatTemplate: chatTemplate,
            hasVision: visionValue != nil
        )
    }

    private static func readStringValue(forKey key: String, from model: OpaquePointer) -> String? {
        var buffer = [CChar](repeating: 0, count: 1024)
        let length = key.withCString { keyPointer in
            buffer.withUnsafeMutableBufferPointer { pointer in
                llama_model_meta_val_str(model, keyPointer, pointer.baseAddress, pointer.count)
            }
        }

        guard length >= 0 else {
            return nil
        }

        return String(cString: buffer)
    }

    /// Provides raw context access for vision embedding evaluation.
    public func withContext<T>(_ body: (OpaquePointer) throws -> T) throws -> T {
        guard let context else {
            throw LlamaError.decodeFailed
        }

        return try body(context)
    }

    public func getEmbedding(tokens: [Int32]) throws -> [Float] {
        guard let model, let context else {
            throw LlamaError.decodeFailed
        }

        llama_set_embeddings(context, true)
        defer { llama_set_embeddings(context, false) }

        llama_kv_cache_clear(context)

        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }

        batch.n_tokens = 0
        for (index, token) in tokens.enumerated() {
            let batchIndex = Int(batch.n_tokens)
            batch.token[batchIndex] = token
            batch.pos[batchIndex] = Int32(index)
            batch.n_seq_id[batchIndex] = 1
            batch.seq_id[batchIndex]![0] = 0
            batch.logits[batchIndex] = 1
            batch.n_tokens += 1
        }

        guard llama_decode(context, batch) == 0 else {
            throw LlamaError.decodeFailed
        }

        let nEmbd = Int(llama_model_n_embd(model))
        var embeddingPointer = llama_get_embeddings_seq(context, 0)
        if embeddingPointer == nil {
            embeddingPointer = llama_get_embeddings_ith(context, -1)
        }
        guard let embeddingPointer else {
            throw LlamaError.decodeFailed
        }

        let raw = Array(UnsafeBufferPointer(start: embeddingPointer, count: nEmbd))
        let norm = sqrt(raw.reduce(Float.zero) { $0 + ($1 * $1) })
        let scale: Float = norm > 1e-12 ? (1.0 / norm) : 0.0
        return raw.map { $0 * scale }
    }

    public var embeddingDims: Int {
        guard let model else {
            return 0
        }

        return Int(llama_model_n_embd(model))
    }
}
