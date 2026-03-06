import Foundation

public enum MLXModelDetector {

    /// Returns true if `path` is an MLX model directory containing
    /// config.json and at least one .safetensors file.
    public static func isMLXModelDirectory(_ path: String) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        let configPath = (path as NSString).appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configPath) else {
            return false
        }
        guard let contents = try? fm.contentsOfDirectory(atPath: path) else {
            return false
        }
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    /// Reads chat_template from tokenizer_config.json if present.
    public static func readChatTemplate(from modelPath: String) -> String? {
        let path = (modelPath as NSString).appendingPathComponent("tokenizer_config.json")
        guard let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let template = json["chat_template"] as? String else {
            return nil
        }
        return template
    }

    /// Reads model name from config.json (_name_or_path or model_type).
    public static func readModelName(from modelPath: String) -> String? {
        guard let json = readConfigJSON(from: modelPath) else { return nil }
        return json["_name_or_path"] as? String ?? json["model_type"] as? String
    }

    /// Reads max_position_embeddings from config.json for context size.
    public static func readMaxPositionEmbeddings(from modelPath: String) -> Int? {
        guard let json = readConfigJSON(from: modelPath) else { return nil }
        return json["max_position_embeddings"] as? Int
    }

    /// Sums the byte size of all files in the model directory.
    public static func directorySize(at path: String) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: path) else { return 0 }
        var total: Int64 = 0
        while let file = enumerator.nextObject() as? String {
            let filePath = (path as NSString).appendingPathComponent(file)
            if let attrs = try? fm.attributesOfItem(atPath: filePath),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }
        return total
    }

    /// Returns true if the model directory contains a VLM (vision_config in config.json).
    public static func isVLMModel(from modelPath: String) -> Bool {
        guard let json = readConfigJSON(from: modelPath) else { return false }
        return json["vision_config"] != nil
    }

    private static func readConfigJSON(from modelPath: String) -> [String: Any]? {
        let path = (modelPath as NSString).appendingPathComponent("config.json")
        guard let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }
}
