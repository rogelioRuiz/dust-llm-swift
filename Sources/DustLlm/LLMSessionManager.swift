import Foundation
import DustCore

public enum MemoryPressureLevel {
    case standard
    case critical
}

public final class LLMSessionManager: DustModelServer, @unchecked Sendable {
    public typealias SessionFactory = @Sendable (_ path: String, _ modelId: String, _ config: LLMConfig, _ priority: DustSessionPriority) throws -> LlamaSession
    public typealias VisionEncoderFactory = @Sendable (_ mmprojPath: String, _ model: OpaquePointer) throws -> VisionEncoderProtocol

    public static let inferenceQueue = DispatchQueue(
        label: "io.t6x.dust.llm.inference",
        qos: .userInitiated
    )

    private let lock = NSLock()
    private let sessionFactory: SessionFactory
    private var descriptors: [String: DustModelDescriptor] = [:]
    private var statuses: [String: DustModelStatus] = [:]
    private var configs: [String: LLMConfig] = [:]
    private var cachedSessions: [String: CachedSession] = [:]

    public init(
        sessionFactory: SessionFactory? = nil,
        visionEncoderFactory: @escaping VisionEncoderFactory = { mmprojPath, model in
            try VisionEncoder(mmprojPath: mmprojPath, model: model)
        }
    ) {
        if let sessionFactory {
            self.sessionFactory = sessionFactory
        } else {
            self.sessionFactory = { path, modelId, config, priority in
                let context = try LlamaContext(path: path, config: config)
                let visionEncoder = try Self.makeVisionEncoder(
                    for: context.metadata,
                    model: context.model,
                    modelPath: path,
                    config: config,
                    visionEncoderFactory: visionEncoderFactory
                )
                return LlamaSession(
                    sessionId: modelId,
                    context: context,
                    priority: priority,
                    visionEncoder: visionEncoder
                )
            }
        }
    }

    public func register(
        descriptor: DustModelDescriptor,
        config: LLMConfig? = nil
    ) {
        let status = initialStatus(for: descriptor)

        lock.lock()
        descriptors[descriptor.id] = descriptor
        statuses[descriptor.id] = status
        configs[descriptor.id] = config ?? configs[descriptor.id] ?? LLMConfig()
        lock.unlock()
    }

    public func setStatus(
        _ status: DustModelStatus,
        for id: String
    ) {
        lock.lock()
        statuses[id] = status
        lock.unlock()
    }

    public func loadModel(
        descriptor: DustModelDescriptor,
        priority: DustSessionPriority
    ) async throws -> any DustModelSession {
        let config: LLMConfig

        lock.lock()
        guard descriptors[descriptor.id] != nil else {
            lock.unlock()
            throw DustCoreError.modelNotFound
        }
        config = configs[descriptor.id] ?? LLMConfig()
        lock.unlock()

        return try loadModelWithConfig(
            descriptor: descriptor,
            config: config,
            priority: priority
        )
    }

    public func loadModelWithConfig(
        descriptor: DustModelDescriptor,
        config: LLMConfig,
        priority: DustSessionPriority
    ) throws -> LlamaSession {
        if let cached = incrementCachedRefCount(for: descriptor.id) {
            return cached
        }

        let registeredDescriptor: DustModelDescriptor
        let status: DustModelStatus

        lock.lock()
        guard let storedDescriptor = descriptors[descriptor.id] else {
            lock.unlock()
            throw DustCoreError.modelNotFound
        }
        registeredDescriptor = storedDescriptor
        status = statuses[descriptor.id] ?? .notLoaded
        lock.unlock()

        guard status == .ready else {
            throw DustCoreError.modelNotReady
        }

        guard let path = resolvedPath(for: registeredDescriptor) else {
            throw DustCoreError.invalidInput(detail: "descriptor.url or descriptor.metadata.localPath is required")
        }

        let createdSession = try sessionFactory(path, descriptor.id, config, priority)

        var installedSession: LlamaSession?
        var discardedSession: LlamaSession?

        lock.lock()
        if var cached = cachedSessions[descriptor.id] {
            cached.refCount += 1
            cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
            cachedSessions[descriptor.id] = cached
            installedSession = cached.session
            discardedSession = createdSession
        } else {
            cachedSessions[descriptor.id] = CachedSession(
                session: createdSession,
                priority: priority,
                refCount: 1,
                lastAccessTime: DispatchTime.now().uptimeNanoseconds
            )
            installedSession = createdSession
        }
        statuses[descriptor.id] = .ready
        configs[descriptor.id] = config
        lock.unlock()

        discardedSession?.evict()
        return installedSession ?? createdSession
    }

    public func loadModel(
        path: String,
        modelId: String,
        config: LLMConfig,
        priority: DustSessionPriority
    ) throws -> LlamaSession {
        let descriptor = legacyDescriptor(path: path, modelId: modelId)
        register(descriptor: descriptor, config: config)
        setStatus(.ready, for: modelId)

        return try loadModelWithConfig(
            descriptor: descriptor,
            config: config,
            priority: priority
        )
    }

    public func unloadModel(id: String) async throws {
        let nextRefCount: Int?

        lock.lock()
        if var cached = cachedSessions[id], cached.refCount > 0 {
            cached.refCount -= 1
            cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
            cachedSessions[id] = cached
            nextRefCount = cached.refCount
        } else {
            nextRefCount = nil
        }
        lock.unlock()

        guard nextRefCount != nil else {
            throw DustCoreError.modelNotFound
        }
    }

    public func forceUnloadModel(id: String) async throws {
        let session: LlamaSession?

        lock.lock()
        session = cachedSessions.removeValue(forKey: id)?.session
        lock.unlock()

        guard let session else {
            throw DustCoreError.modelNotFound
        }

        try await session.close()
    }

    public func evict(modelId: String) async -> LlamaSession? {
        let session: LlamaSession?

        lock.lock()
        session = cachedSessions.removeValue(forKey: modelId)?.session
        lock.unlock()

        session?.evict()
        return session
    }

    public func evictUnderPressure(level: MemoryPressureLevel) async {
        let evicted: [LlamaSession]

        lock.lock()
        let eligible = cachedSessions.filter { (_, cached) in
            guard cached.refCount == 0 else {
                return false
            }

            switch level {
            case .standard:
                return cached.priority == .background
            case .critical:
                return true
            }
        }
        let sorted = eligible.sorted { $0.value.lastAccessTime < $1.value.lastAccessTime }
        evicted = sorted.map(\.value.session)
        for (id, _) in sorted {
            cachedSessions.removeValue(forKey: id)
        }
        lock.unlock()

        for session in evicted {
            session.evict()
        }
    }

    public func listModels() async throws -> [DustModelDescriptor] {
        allDescriptors()
    }

    public func modelStatus(id: String) async throws -> DustModelStatus {
        lock.lock()
        defer { lock.unlock() }
        return statuses[id] ?? .notLoaded
    }

    public func refCount(for id: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions[id]?.refCount ?? 0
    }

    public func hasCachedSession(for id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions[id] != nil
    }

    public func session(for id: String) -> LlamaSession? {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions[id]?.session
    }

    public func allModelIds() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.keys.sorted()
    }

    public func allDescriptors() -> [DustModelDescriptor] {
        lock.lock()
        defer { lock.unlock() }
        return descriptors.values.sorted { $0.id < $1.id }
    }

    public var sessionCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedSessions.count
    }

    private func incrementCachedRefCount(for id: String) -> LlamaSession? {
        let cachedSession: LlamaSession?

        lock.lock()
        if var cached = cachedSessions[id] {
            cached.refCount += 1
            cached.lastAccessTime = DispatchTime.now().uptimeNanoseconds
            cachedSessions[id] = cached
            cachedSession = cached.session
        } else {
            cachedSession = nil
        }
        lock.unlock()

        return cachedSession
    }

    private func initialStatus(for descriptor: DustModelDescriptor) -> DustModelStatus {
        guard let path = resolvedPath(for: descriptor) else {
            return .notLoaded
        }

        return FileManager.default.fileExists(atPath: path) ? .ready : .notLoaded
    }

    private func resolvedPath(for descriptor: DustModelDescriptor) -> String? {
        if let localPath = descriptor.metadata?["localPath"], !localPath.isEmpty {
            return localPath
        }

        if let url = descriptor.url, !url.isEmpty {
            return url
        }

        return nil
    }

    private func legacyDescriptor(
        path: String,
        modelId: String
    ) -> DustModelDescriptor {
        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let sizeBytes = (attributes?[.size] as? NSNumber)?.int64Value ?? 0

        return DustModelDescriptor(
            id: modelId,
            name: modelId,
            format: .gguf,
            sizeBytes: sizeBytes,
            version: "legacy",
            url: path
        )
    }

    private static func makeVisionEncoder(
        for metadata: LLMModelMetadata,
        model: OpaquePointer?,
        modelPath: String,
        config: LLMConfig,
        visionEncoderFactory: VisionEncoderFactory
    ) throws -> VisionEncoderProtocol? {
        guard metadata.hasVision, let model else {
            return nil
        }

        return try visionEncoderFactory(resolveMMProjPath(modelPath: modelPath, config: config), model)
    }

    private static func resolveMMProjPath(
        modelPath: String,
        config: LLMConfig
    ) -> String {
        if let mmprojPath = config.mmprojPath, !mmprojPath.isEmpty {
            return mmprojPath
        }

        guard modelPath.hasSuffix(".gguf") else {
            return modelPath + "-mmproj.gguf"
        }

        return String(modelPath.dropLast(".gguf".count)) + "-mmproj.gguf"
    }
}

private struct CachedSession {
    let session: LlamaSession
    let priority: DustSessionPriority
    var refCount: Int
    var lastAccessTime: UInt64
}
