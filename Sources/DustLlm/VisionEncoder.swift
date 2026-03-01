import Foundation
import DustCore
import llama
import llava

public struct ImageEmbedding {
    public let handle: UnsafeMutablePointer<llava_image_embed>

    public var tokenCount: Int {
        Int(handle.pointee.n_image_pos)
    }
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
    private var clipContext: OpaquePointer?

    public init(mmprojPath: String) throws {
        guard FileManager.default.fileExists(atPath: mmprojPath) else {
            throw LlamaError.fileNotFound(path: mmprojPath)
        }

        clipContext = mmprojPath.withCString {
            clip_model_load($0, 1)
        }

        guard clipContext != nil else {
            throw LlamaError.loadFailed(path: mmprojPath)
        }
    }

    public var imageTokenCount: Int {
        guard let clipContext else {
            return 0
        }

        return Int(clip_n_patches(clipContext)) + 1
    }

    public func encode(imageBytes: Data) throws -> ImageEmbedding {
        guard let clipContext else {
            throw LlamaError.modelEvicted
        }

        guard !imageBytes.isEmpty else {
            throw DustCoreError.inferenceFailed(detail: "Failed to encode image")
        }

        let handle = imageBytes.withUnsafeBytes { rawBuffer -> UnsafeMutablePointer<llava_image_embed>? in
            guard let baseAddress = rawBuffer.baseAddress else {
                return nil
            }

            return llava_image_embed_make_with_bytes(
                clipContext,
                4,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                Int32(imageBytes.count)
            )
        }

        guard let handle else {
            throw DustCoreError.inferenceFailed(detail: "Failed to encode image")
        }

        return ImageEmbedding(handle: handle)
    }

    public func evalImageEmbed(
        embedding: ImageEmbedding,
        context: OpaquePointer,
        nPast: inout Int32
    ) throws {
        let batchSize = Int32(llama_n_batch(context))
        let ok = llava_eval_image_embed(context, embedding.handle, batchSize, &nPast)
        guard ok else {
            throw DustCoreError.inferenceFailed(detail: "Failed to evaluate image embedding")
        }
    }

    public func freeEmbedding(_ embedding: ImageEmbedding) {
        llava_image_embed_free(embedding.handle)
    }

    public func close() {
        if let clipContext {
            clip_free(clipContext)
            self.clipContext = nil
        }
    }

    deinit {
        close()
    }
}
