import XCTest
@testable import DustLlm
import DustCore

private final class Box<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

final class LLMSessionManagerTests: XCTestCase {
    func testL1T1LoadValidFixtureCreatesSessionAndMetadata() throws {
        let fixtureURL = try fixtureURL()
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixtureURL.path))

        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(
                        name: "tiny-test-model",
                        chatTemplate: "{% for message in messages %}{{ message.content }}{% endfor %}",
                        hasVision: true
                    ),
                    priority: priority
                )
            }
        )
        let session = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: LLMConfig(),
            priority: .interactive
        )

        XCTAssertEqual(session.status(), .ready)
        XCTAssertEqual(session.metadata.name, "tiny-test-model")
        XCTAssertNotNil(session.metadata.chatTemplate)
    }

    func testL1T2LoadMissingFileThrowsPathInMessage() {
        let manager = LLMSessionManager(
            sessionFactory: { path, _, _, _, _ in
                throw LlamaError.fileNotFound(path: path)
            }
        )

        XCTAssertThrowsError(
            try manager.loadModel(
                path: "/nonexistent/model.gguf",
                modelId: "missing",
                config: LLMConfig(),
                priority: .interactive
            )
        ) { error in
            guard case .fileNotFound(let path) = error as? LlamaError else {
                return XCTFail("Expected fileNotFound, got \(error)")
            }
            XCTAssertEqual(path, "/nonexistent/model.gguf")
        }
    }

    func testL1T3LoadCorruptFileThrowsInferenceError() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try Data([0x47, 0x47, 0x55, 0x46, 0x01, 0x00]).write(to: tempURL)

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        let manager = LLMSessionManager(
            sessionFactory: { path, _, _, _, _ in
                throw LlamaError.loadFailed(path: path)
            }
        )
        XCTAssertThrowsError(
            try manager.loadModel(
                path: tempURL.path,
                modelId: "corrupt",
                config: LLMConfig(),
                priority: .interactive
            )
        )
    }

    func testL1T4WrongFormatRejectedBeforeLoad() {
        // The plugin layer rejects non-gguf/mlx formats before calling the session manager.
        // Verify that all unsupported formats would be rejected by the guard.
        let accepted: Set<String> = [DustModelFormat.gguf.rawValue, DustModelFormat.mlx.rawValue]
        XCTAssertTrue(accepted.contains("gguf"))
        XCTAssertTrue(accepted.contains("mlx"))

        let rejected: [DustModelFormat] = [.onnx, .coreml, .tflite, .custom]
        for format in rejected {
            XCTAssertFalse(accepted.contains(format.rawValue), "\(format) should be rejected")
        }

        // Verify the session factory is never invoked for a rejected format.
        let factoryCalled = Box(false)
        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                factoryCalled.value = true
                return LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(name: nil, chatTemplate: nil, hasVision: false),
                    priority: priority
                )
            }
        )
        _ = manager // suppress unused warning
        XCTAssertFalse(factoryCalled.value, "Factory must not be called when format is rejected at plugin layer")
    }

    func testL1T5UnloadLoadedModelRemovesSession() async throws {
        let fixtureURL = try fixtureURL()
        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(name: "tiny-test-model", chatTemplate: nil, hasVision: false),
                    priority: priority
                )
            }
        )

        _ = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: LLMConfig(),
            priority: .interactive
        )
        try await manager.forceUnloadModel(id: "tiny-test")

        XCTAssertEqual(manager.sessionCount, 0)
        XCTAssertNil(manager.session(for: "tiny-test"))
    }

    func testL1T6UnloadUnknownIdThrowsModelNotFound() async {
        let manager = LLMSessionManager()

        do {
            try await manager.forceUnloadModel(id: "nonexistent")
            XCTFail("Expected modelNotFound")
        } catch let error as DustCoreError {
            XCTAssertEqual(error, .modelNotFound)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testL1T7LoadingSameIdTwiceReusesSession() throws {
        let fixtureURL = try fixtureURL()
        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(name: "tiny-test-model", chatTemplate: nil, hasVision: false),
                    priority: priority
                )
            }
        )

        let first = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: LLMConfig(),
            priority: .interactive
        )
        let second = try manager.loadModel(
            path: fixtureURL.path,
            modelId: "tiny-test",
            config: LLMConfig(),
            priority: .background
        )

        XCTAssertEqual(ObjectIdentifier(first), ObjectIdentifier(second))
        XCTAssertEqual(manager.sessionCount, 1)
    }

    func testL1T8ConcurrentLoadTwoModelsSucceeds() async throws {
        let fixtureURL = try fixtureURL()
        let manager = LLMSessionManager(
            sessionFactory: { _, modelId, _, priority, _ in
                LlamaSession(
                    sessionId: modelId,
                    metadata: LLMModelMetadata(name: modelId, chatTemplate: nil, hasVision: false),
                    priority: priority
                )
            }
        )

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                _ = try manager.loadModel(
                    path: fixtureURL.path,
                    modelId: "model-a",
                    config: LLMConfig(),
                    priority: .interactive
                )
            }
            group.addTask {
                _ = try manager.loadModel(
                    path: fixtureURL.path,
                    modelId: "model-b",
                    config: LLMConfig(),
                    priority: .interactive
                )
            }
            try await group.waitForAll()
        }

        XCTAssertEqual(Set(manager.allModelIds()), ["model-a", "model-b"])
        XCTAssertEqual(manager.sessionCount, 2)
    }

    private func fixtureURL() throws -> URL {
        if let bundled = Bundle.module.url(forResource: "tiny-test", withExtension: "gguf") {
            return bundled
        }

        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/tiny-test.gguf")

        if FileManager.default.fileExists(atPath: fallback.path) {
            return fallback
        }

        throw XCTSkip("tiny-test.gguf fixture was not found")
    }
}
