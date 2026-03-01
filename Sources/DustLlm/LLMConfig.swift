import Foundation

public struct LLMConfig: Equatable, Sendable {
    public let nGpuLayers: Int32
    public let contextSize: UInt32
    public let batchSize: UInt32
    public let mmprojPath: String?

    public init(
        nGpuLayers: Int32 = -1,
        contextSize: UInt32 = 2048,
        batchSize: UInt32 = 512,
        mmprojPath: String? = nil
    ) {
        self.nGpuLayers = nGpuLayers
        self.contextSize = contextSize
        self.batchSize = batchSize
        self.mmprojPath = mmprojPath
    }

    public init(jsObject: [String: Any]?) {
        self.init(
            nGpuLayers: Int32(jsObject?["nGpuLayers"] as? Int ?? -1),
            contextSize: UInt32(jsObject?["contextSize"] as? Int ?? 2048),
            batchSize: UInt32(jsObject?["batchSize"] as? Int ?? 512),
            mmprojPath: jsObject?["mmprojPath"] as? String
        )
    }
}
