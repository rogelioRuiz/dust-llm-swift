import XCTest
@testable import DustLlm
import DustCore

final class LLMRegistryTests: XCTestCase {
    override func setUp() {
        super.setUp()
        DustCoreRegistry.shared.resetForTesting()
    }

    override func tearDown() {
        DustCoreRegistry.shared.resetForTesting()
        super.tearDown()
    }

    func testL5T1RegistryRegistrationMakesManagerResolvable() throws {
        let manager = makeManager()

        DustCoreRegistry.shared.register(modelServer: manager)

        let resolved = try DustCoreRegistry.shared.resolveModelServer()
        XCTAssertTrue((resolved as AnyObject) === manager)
    }

    func testL5T2LoadModelForReadyDescriptorCreatesSessionAndRefCount() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let session = try await manager.loadModel(descriptor: descriptor, priority: .interactive)

        XCTAssertEqual(session.status(), .ready)
        XCTAssertEqual(manager.refCount(for: "model-a"), 1)
    }

    func testL5T3LoadModelForNotLoadedDescriptorThrowsModelNotReady() async throws {
        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: "/tmp/missing-model.gguf")
        manager.register(descriptor: descriptor)

        do {
            _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
            XCTFail("Expected modelNotReady")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotReady)
        }
    }

    func testL5T4LoadModelForUnregisteredIdThrowsModelNotFound() async {
        let manager = makeManager()
        let descriptor = makeDescriptor(id: "ghost", path: "/tmp/ghost.gguf")

        do {
            _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
            XCTFail("Expected modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testL5T5UnloadModelDecrementsRefCountAndKeepsSessionCached() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        _ = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        try await manager.unloadModel(id: "model-a")

        XCTAssertEqual(manager.refCount(for: "model-a"), 0)
        XCTAssertTrue(manager.hasCachedSession(for: "model-a"))
    }

    func testL5T6LoadModelTwiceReusesSameSessionAndIncrementsRefCount() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let first = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        let second = try await manager.loadModel(descriptor: descriptor, priority: .background)

        XCTAssertEqual(identity(of: first), identity(of: second))
        XCTAssertEqual(manager.refCount(for: "model-a"), 2)
    }

    func testL5T7EvictOnZeroRefCountSessionInvalidatesSession() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager()
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let loaded = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        let session = try XCTUnwrap(loaded as? LlamaSession)
        try await manager.unloadModel(id: "model-a")

        _ = await manager.evict(modelId: "model-a")

        XCTAssertTrue(session.isModelEvicted)
        XCTAssertFalse(manager.hasCachedSession(for: "model-a"))
    }

    func testL5T8GenerateOnEvictedSessionThrowsModelEvicted() async throws {
        let fileURL = try makeTempModelFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let manager = makeManager(engineFactory: { priority in
            LlamaSession(
                sessionId: "model-a",
                engine: RegistryMockLlamaEngine(),
                metadata: LLMModelMetadata(name: "mock", chatTemplate: nil, hasVision: false),
                priority: priority
            )
        })
        let descriptor = makeDescriptor(id: "model-a", path: fileURL.path)
        manager.register(descriptor: descriptor)

        let loaded = try await manager.loadModel(descriptor: descriptor, priority: .interactive)
        let session = try XCTUnwrap(loaded as? LlamaSession)
        try await manager.unloadModel(id: "model-a")
        _ = await manager.evict(modelId: "model-a")

        XCTAssertThrowsError(
            try session.generate(
                prompt: "prompt",
                maxTokens: 1,
                stopSequences: [],
                sampler: SamplerConfig()
            )
        ) { error in
            XCTAssertEqual(error as? LlamaError, .modelEvicted)
        }
    }

    func testL5T9AllModelIdsReturnsOnlyLiveSessionsAfterEviction() async throws {
        let fileA = try makeTempModelFile()
        let fileB = try makeTempModelFile()
        defer {
            try? FileManager.default.removeItem(at: fileA)
            try? FileManager.default.removeItem(at: fileB)
        }

        let manager = makeManager()
        let descriptorA = makeDescriptor(id: "model-a", path: fileA.path)
        let descriptorB = makeDescriptor(id: "model-b", path: fileB.path)
        manager.register(descriptor: descriptorA)
        manager.register(descriptor: descriptorB)

        _ = try await manager.loadModel(descriptor: descriptorA, priority: .interactive)
        _ = try await manager.loadModel(descriptor: descriptorB, priority: .interactive)

        XCTAssertEqual(manager.allModelIds(), ["model-a", "model-b"])

        try await manager.unloadModel(id: "model-a")
        _ = await manager.evict(modelId: "model-a")

        XCTAssertEqual(manager.allModelIds(), ["model-b"])
    }

    private func makeManager(
        engineFactory: ((DustSessionPriority) -> LlamaSession)? = nil
    ) -> LLMSessionManager {
        LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                if let engineFactory {
                    return engineFactory(priority)
                }

                return LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(name: modelId, chatTemplate: nil, hasVision: false),
                    priority: priority
                )
            }
        )
    }

    private func makeDescriptor(
        id: String,
        path: String
    ) -> DustModelDescriptor {
        DustModelDescriptor(
            id: id,
            name: id,
            format: .gguf,
            sizeBytes: 1,
            version: "1.0.0",
            metadata: ["localPath": path]
        )
    }

    private func makeTempModelFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try Data([0x47, 0x47, 0x55, 0x46]).write(to: url)
        return url
    }
}

private func identity(of session: any DustModelSession) -> ObjectIdentifier {
    ObjectIdentifier(session as AnyObject)
}

private final class RegistryMockLlamaEngine: LlamaEngine, @unchecked Sendable {
    let nCtx: UInt32 = 64

    func tokenize(text: String, addSpecial: Bool) throws -> [Int32] {
        _ = text
        _ = addSpecial
        return [1]
    }

    func detokenize(tokens: [Int32]) throws -> String {
        _ = tokens
        return "mock"
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
        _ = promptTokens
        _ = maxTokens
        _ = sampler
        return ([2], .eos)
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
        return .eos
    }
}
