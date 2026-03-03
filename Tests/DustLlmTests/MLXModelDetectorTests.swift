import XCTest
@testable import DustLlm

final class MLXModelDetectorTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXModelDetectorTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    // MARK: - isMLXModelDirectory

    func testGGUFFileIsNotMLXModel() {
        let ggufPath = tempDir.appendingPathComponent("model.gguf").path
        FileManager.default.createFile(atPath: ggufPath, contents: Data([0x00]))
        XCTAssertFalse(MLXModelDetector.isMLXModelDirectory(ggufPath))
    }

    func testNonexistentPathIsNotMLXModel() {
        XCTAssertFalse(MLXModelDetector.isMLXModelDirectory("/nonexistent/path"))
    }

    func testDirectoryWithoutConfigIsNotMLX() {
        let safetensorsPath = tempDir.appendingPathComponent("model.safetensors").path
        FileManager.default.createFile(atPath: safetensorsPath, contents: Data([0x00]))
        XCTAssertFalse(MLXModelDetector.isMLXModelDirectory(tempDir.path))
    }

    func testDirectoryWithoutSafetensorsIsNotMLX() {
        let configPath = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: configPath, contents: "{}".data(using: .utf8))
        XCTAssertFalse(MLXModelDetector.isMLXModelDirectory(tempDir.path))
    }

    func testValidMLXDirectoryDetected() {
        let configPath = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: configPath, contents: "{}".data(using: .utf8))
        let safetensorsPath = tempDir.appendingPathComponent("model.safetensors").path
        FileManager.default.createFile(atPath: safetensorsPath, contents: Data([0x00]))
        XCTAssertTrue(MLXModelDetector.isMLXModelDirectory(tempDir.path))
    }

    // MARK: - readChatTemplate

    func testReadChatTemplateFromTokenizerConfig() {
        let template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        let json: [String: Any] = ["chat_template": template]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("tokenizer_config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertEqual(MLXModelDetector.readChatTemplate(from: tempDir.path), template)
    }

    func testReadChatTemplateReturnsNilWhenMissing() {
        XCTAssertNil(MLXModelDetector.readChatTemplate(from: tempDir.path))
    }

    func testReadChatTemplateReturnsNilWhenKeyMissing() {
        let json: [String: Any] = ["model_type": "llama"]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("tokenizer_config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertNil(MLXModelDetector.readChatTemplate(from: tempDir.path))
    }

    // MARK: - readModelName

    func testReadModelNameFromNameOrPath() {
        let json: [String: Any] = ["_name_or_path": "Qwen/Qwen2.5-0.5B"]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertEqual(MLXModelDetector.readModelName(from: tempDir.path), "Qwen/Qwen2.5-0.5B")
    }

    func testReadModelNameFallsBackToModelType() {
        let json: [String: Any] = ["model_type": "qwen2"]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertEqual(MLXModelDetector.readModelName(from: tempDir.path), "qwen2")
    }

    // MARK: - readMaxPositionEmbeddings

    func testReadMaxPositionEmbeddings() {
        let json: [String: Any] = ["max_position_embeddings": 32768]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertEqual(MLXModelDetector.readMaxPositionEmbeddings(from: tempDir.path), 32768)
    }

    func testReadMaxPositionEmbeddingsReturnsNilWhenMissing() {
        let json: [String: Any] = ["model_type": "llama"]
        let data = try! JSONSerialization.data(withJSONObject: json)
        let path = tempDir.appendingPathComponent("config.json").path
        FileManager.default.createFile(atPath: path, contents: data)

        XCTAssertNil(MLXModelDetector.readMaxPositionEmbeddings(from: tempDir.path))
    }

    // MARK: - directorySize

    func testDirectorySizeSumsFiles() {
        let file1 = tempDir.appendingPathComponent("a.bin").path
        let file2 = tempDir.appendingPathComponent("b.bin").path
        FileManager.default.createFile(atPath: file1, contents: Data(repeating: 0, count: 100))
        FileManager.default.createFile(atPath: file2, contents: Data(repeating: 0, count: 200))

        XCTAssertEqual(MLXModelDetector.directorySize(at: tempDir.path), 300)
    }

    func testDirectorySizeReturnsZeroForNonexistent() {
        XCTAssertEqual(MLXModelDetector.directorySize(at: "/nonexistent"), 0)
    }
}
