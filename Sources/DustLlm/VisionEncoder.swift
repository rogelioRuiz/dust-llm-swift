import Foundation
import DustCore
import llama
import mtmd

public struct ImageEmbedding {
    public let chunks: OpaquePointer  // mtmd_input_chunks *
    public let tokenCount: Int
}

public protocol VisionEncoderProtocol: AnyObject {
    var imageTokenCount: Int { get }
    func encode(imageBytes: Data) throws -> ImageEmbedding
    func evalImageEmbed(
        embedding: ImageEmbedding,
        context: OpaquePointer,
        nPast: inout Int32
    ) throws
    func freeEmbedding(_ embedding: ImageEmbedding)
    func close()
}

public final class VisionEncoder: VisionEncoderProtocol, @unchecked Sendable {
    private var mtmdContext: OpaquePointer?  // mtmd_context *

    public init(mmprojPath: String, model: OpaquePointer) throws {
        guard FileManager.default.fileExists(atPath: mmprojPath) else {
            throw LlamaError.fileNotFound(path: mmprojPath)
        }

        var params = mtmd_context_params_default()
        params.use_gpu = true

        mtmdContext = mmprojPath.withCString { pathPtr in
            mtmd_init_from_file(pathPtr, model, params)
        }

        guard mtmdContext != nil else {
            throw LlamaError.loadFailed(path: mmprojPath)
        }
    }

    public var imageTokenCount: Int {
        // With mtmd, token count is determined per-image during tokenization.
        // Return 0 as a placeholder; actual count comes from the embedding.
        0
    }

    public func encode(imageBytes: Data) throws -> ImageEmbedding {
        guard let mtmdContext else {
            throw LlamaError.modelEvicted
        }

        guard !imageBytes.isEmpty else {
            throw DustCoreError.inferenceFailed(detail: "Failed to encode image")
        }

        let bitmap: OpaquePointer? = imageBytes.withUnsafeBytes { rawBuffer -> OpaquePointer? in
            guard let baseAddress = rawBuffer.baseAddress else {
                return nil
            }
            return mtmd_helper_bitmap_init_from_buf(
                mtmdContext,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                imageBytes.count
            )
        }

        guard let bitmap else {
            throw DustCoreError.inferenceFailed(detail: "Failed to decode image bitmap")
        }

        guard let chunks = mtmd_input_chunks_init() else {
            mtmd_bitmap_free(bitmap)
            throw DustCoreError.inferenceFailed(detail: "Failed to create input chunks")
        }

        let marker = String(cString: mtmd_default_marker())
        var inputText = mtmd_input_text(
            text: nil,
            add_special: true,
            parse_special: true
        )

        let result: Int32 = marker.withCString { markerPtr in
            inputText.text = markerPtr
            var bitmapPtr: OpaquePointer? = bitmap
            return withUnsafeMutablePointer(to: &bitmapPtr) { bitmapPtrPtr in
                mtmd_tokenize(
                    mtmdContext,
                    chunks,
                    &inputText,
                    bitmapPtrPtr,
                    1
                )
            }
        }

        mtmd_bitmap_free(bitmap)

        guard result == 0 else {
            mtmd_input_chunks_free(chunks)
            throw DustCoreError.inferenceFailed(detail: "Failed to tokenize image (error \(result))")
        }

        let tokenCount = Int(mtmd_helper_get_n_tokens(chunks))
        return ImageEmbedding(chunks: chunks, tokenCount: tokenCount)
    }

    public func evalImageEmbed(
        embedding: ImageEmbedding,
        context: OpaquePointer,
        nPast: inout Int32
    ) throws {
        guard let mtmdContext else {
            throw LlamaError.modelEvicted
        }

        let batchSize = Int32(llama_n_batch(context))
        var newNPast = nPast
        let result = mtmd_helper_eval_chunks(
            mtmdContext,
            context,
            embedding.chunks,
            nPast,
            0,           // seq_id
            batchSize,
            true,        // logits_last
            &newNPast
        )

        guard result == 0 else {
            throw DustCoreError.inferenceFailed(detail: "Failed to evaluate image embedding")
        }

        nPast = newNPast
    }

    public func freeEmbedding(_ embedding: ImageEmbedding) {
        mtmd_input_chunks_free(embedding.chunks)
    }

    public func close() {
        if let mtmdContext {
            mtmd_free(mtmdContext)
            self.mtmdContext = nil
        }
    }

    deinit {
        close()
    }
}
