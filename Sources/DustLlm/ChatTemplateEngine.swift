import Foundation

public struct ChatMessage: Equatable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public final class ChatTemplateEngine {
    public static let chatMLTemplate = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    private let templateString: String

    public init(templateString: String?) {
        self.templateString = templateString ?? Self.chatMLTemplate
    }

    public func apply(
        messages: [ChatMessage],
        addGenerationPrompt: Bool,
        bosToken: String = "",
        eosToken: String = "",
        enableThinking: Bool? = nil
    ) throws -> String {
        let parser = ChatTemplateParser(template: templateString)
        let nodes = try parser.parse()
        var rootContext: [String: Any] = [
            "messages": messages.map { message in
                [
                    "role": message.role,
                    "content": message.content,
                ] as [String: Any] as Any
            },
            "add_generation_prompt": addGenerationPrompt,
            "bos_token": bosToken,
            "eos_token": eosToken,
        ]
        if let enableThinking {
            rootContext["enable_thinking"] = enableThinking
        }

        return try ChatTemplateEvaluator(rootContext: rootContext).render(nodes)
    }
}

private enum ChatTemplateError: Error, LocalizedError {
    case syntax(String)
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case .syntax(let detail):
            return "Chat template syntax error: \(detail)"
        case .runtime(let detail):
            return "Chat template runtime error: \(detail)"
        }
    }
}

private final class ChatTemplateUndefined: @unchecked Sendable {
    static let shared = ChatTemplateUndefined()
    private init() {}
}

private final class ChatTemplateNamespace: @unchecked Sendable {
    var values: [String: Any]

    init(values: [String: Any] = [:]) {
        self.values = values
    }
}

private final class ChatTemplateMacro: @unchecked Sendable {
    let params: [ChatTemplateMacroParam]
    let body: [ChatTemplateNode]

    init(params: [ChatTemplateMacroParam], body: [ChatTemplateNode]) {
        self.params = params
        self.body = body
    }
}

private struct ChatTemplateIfClause {
    let condition: ChatTemplateExpression?
    let body: [ChatTemplateNode]
}

private enum ChatTemplateCallArgument {
    case positional(ChatTemplateExpression)
    case named(String, ChatTemplateExpression)
}

private enum ChatTemplateLiteral {
    case string(String)
    case int(Int)
    case bool(Bool)
}

private enum ChatTemplateUnaryOperator {
    case not
    case negate
}

private enum ChatTemplateBinaryOperator {
    case add
    case equals
    case notEquals
    case and
    case or
    case contains
}

private indirect enum ChatTemplateExpression {
    case literal(ChatTemplateLiteral)
    case variable(String)
    case unary(ChatTemplateUnaryOperator, ChatTemplateExpression)
    case binary(ChatTemplateBinaryOperator, ChatTemplateExpression, ChatTemplateExpression)
    case isDefined(ChatTemplateExpression, negated: Bool)
    case isTest(ChatTemplateExpression, String, negated: Bool)
    case member(ChatTemplateExpression, String)
    case subscriptAccess(ChatTemplateExpression, ChatTemplateExpression)
    case slice(ChatTemplateExpression, start: ChatTemplateExpression?)
    case call(ChatTemplateExpression, [ChatTemplateCallArgument])
    case filter(ChatTemplateExpression, String)
}

private struct ChatTemplateMacroParam {
    let name: String
    let defaultValue: ChatTemplateExpression?
}

private indirect enum ChatTemplateNode {
    case text(String)
    case output(ChatTemplateExpression)
    case forLoop(variable: String, iterable: ChatTemplateExpression, body: [ChatTemplateNode], elseBody: [ChatTemplateNode]?)
    case conditional([ChatTemplateIfClause])
    case setVariable(name: String, value: ChatTemplateExpression)
    case setAttribute(target: String, attribute: String, value: ChatTemplateExpression)
    case macro(name: String, params: [ChatTemplateMacroParam], body: [ChatTemplateNode])
}

private final class ChatTemplateParser {
    private let characters: [Character]
    private var index = 0
    private var trimNextLeadingWhitespace = false

    init(template: String) {
        self.characters = Array(template)
    }

    func parse() throws -> [ChatTemplateNode] {
        let (nodes, _) = try parseNodes(until: [])
        return nodes
    }

    private func parseNodes(until terminators: Set<String>) throws -> ([ChatTemplateNode], String?) {
        var nodes: [ChatTemplateNode] = []

        while index < characters.count {
            if matches("{{", at: index) {
                let tag = try readTag(open: "{{", close: "}}")
                if tag.trimLeft {
                    trimTrailingWhitespace(in: &nodes)
                }

                let expression = try ChatTemplateExpressionParser(source: tag.content).parse()
                nodes.append(.output(expression))
                trimNextLeadingWhitespace = tag.trimRight
                continue
            }

            if matches("{%", at: index) {
                let tag = try readTag(open: "{%", close: "%}")
                if tag.trimLeft {
                    trimTrailingWhitespace(in: &nodes)
                }

                trimNextLeadingWhitespace = tag.trimRight

                let statement = tag.content.trimmingCharacters(in: .whitespacesAndNewlines)
                let head = statementHead(for: statement)

                if terminators.contains(head) {
                    return (nodes, statement)
                }

                switch head {
                case "for":
                    let node = try parseForStatement(statement)
                    nodes.append(node)
                case "if":
                    let node = try parseIfStatement(statement)
                    nodes.append(node)
                case "set":
                    let node = try parseSetStatement(statement)
                    nodes.append(node)
                case "macro":
                    let node = try parseMacroStatement(statement)
                    nodes.append(node)
                default:
                    throw ChatTemplateError.syntax("Unsupported statement: \(statement)")
                }
                continue
            }

            var text = readText()
            if trimNextLeadingWhitespace {
                text = trimLeadingWhitespace(from: text)
                trimNextLeadingWhitespace = false
            }
            if !text.isEmpty {
                nodes.append(.text(text))
            }
        }

        if !terminators.isEmpty {
            throw ChatTemplateError.syntax("Missing terminator: \(terminators.sorted().joined(separator: ", "))")
        }

        return (nodes, nil)
    }

    private func parseForStatement(_ statement: String) throws -> ChatTemplateNode {
        let remainder = String(statement.dropFirst("for".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let (variable, iterableSource) = splitTopLevel(remainder, separator: " in ") else {
            throw ChatTemplateError.syntax("Invalid for statement: \(statement)")
        }

        let variableName = variable.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isValidIdentifier(variableName) else {
            throw ChatTemplateError.syntax("Invalid loop variable: \(variableName)")
        }

        let iterable = try ChatTemplateExpressionParser(source: iterableSource).parse()
        let (body, terminator) = try parseNodes(until: ["else", "endfor"])

        var elseBody: [ChatTemplateNode]?
        if let terminator, statementHead(for: terminator) == "else" {
            let (parsedElseBody, endTerminator) = try parseNodes(until: ["endfor"])
            guard let endTerminator, statementHead(for: endTerminator) == "endfor" else {
                throw ChatTemplateError.syntax("Missing endfor")
            }
            elseBody = parsedElseBody
        } else if terminator == nil {
            throw ChatTemplateError.syntax("Missing endfor")
        }

        return .forLoop(variable: variableName, iterable: iterable, body: body, elseBody: elseBody)
    }

    private func parseIfStatement(_ statement: String) throws -> ChatTemplateNode {
        let initialConditionSource = String(statement.dropFirst("if".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        var clauses: [ChatTemplateIfClause] = []
        var condition = try ChatTemplateExpressionParser(source: initialConditionSource).parse()

        while true {
            let (body, terminator) = try parseNodes(until: ["elif", "else", "endif"])
            clauses.append(ChatTemplateIfClause(condition: condition, body: body))

            guard let terminator else {
                throw ChatTemplateError.syntax("Missing endif")
            }

            let head = statementHead(for: terminator)
            if head == "elif" {
                let nextConditionSource = String(terminator.dropFirst("elif".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                condition = try ChatTemplateExpressionParser(source: nextConditionSource).parse()
                continue
            }

            if head == "else" {
                let (elseBody, endTerminator) = try parseNodes(until: ["endif"])
                guard let endTerminator, statementHead(for: endTerminator) == "endif" else {
                    throw ChatTemplateError.syntax("Missing endif")
                }
                clauses.append(ChatTemplateIfClause(condition: nil, body: elseBody))
            }

            break
        }

        return .conditional(clauses)
    }

    private func parseSetStatement(_ statement: String) throws -> ChatTemplateNode {
        let remainder = String(statement.dropFirst("set".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let (targetSource, valueSource) = splitTopLevel(remainder, separator: " = ") else {
            throw ChatTemplateError.syntax("Invalid set statement: \(statement)")
        }

        let target = targetSource.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = try ChatTemplateExpressionParser(source: valueSource).parse()

        if let dotIndex = target.firstIndex(of: ".") {
            let namespace = String(target[..<dotIndex])
            let attribute = String(target[target.index(after: dotIndex)...])
            guard isValidIdentifier(namespace), isValidIdentifier(attribute) else {
                throw ChatTemplateError.syntax("Invalid namespace assignment: \(target)")
            }
            return .setAttribute(target: namespace, attribute: attribute, value: value)
        }

        guard isValidIdentifier(target) else {
            throw ChatTemplateError.syntax("Invalid variable assignment: \(target)")
        }

        return .setVariable(name: target, value: value)
    }

    private func parseMacroStatement(_ statement: String) throws -> ChatTemplateNode {
        let remainder = String(statement.dropFirst("macro".count)).trimmingCharacters(in: .whitespacesAndNewlines)

        guard let parenOpen = remainder.firstIndex(of: "(") else {
            throw ChatTemplateError.syntax("Invalid macro statement: \(statement)")
        }
        guard remainder.last == ")" else {
            throw ChatTemplateError.syntax("Invalid macro statement: \(statement)")
        }

        let name = String(remainder[..<parenOpen]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard isValidIdentifier(name) else {
            throw ChatTemplateError.syntax("Invalid macro name: \(name)")
        }

        let paramsStart = remainder.index(after: parenOpen)
        let paramsEnd = remainder.index(before: remainder.endIndex)
        let paramsString = String(remainder[paramsStart..<paramsEnd]).trimmingCharacters(in: .whitespacesAndNewlines)

        var params: [ChatTemplateMacroParam] = []
        if !paramsString.isEmpty {
            let paramParts = splitMacroParams(paramsString)
            for part in paramParts {
                let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
                if let equalsIndex = trimmed.firstIndex(of: "=") {
                    let paramName = String(trimmed[..<equalsIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let defaultSource = String(trimmed[trimmed.index(after: equalsIndex)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    guard isValidIdentifier(paramName) else {
                        throw ChatTemplateError.syntax("Invalid macro parameter: \(paramName)")
                    }
                    let defaultExpr = try ChatTemplateExpressionParser(source: defaultSource).parse()
                    params.append(ChatTemplateMacroParam(name: paramName, defaultValue: defaultExpr))
                } else {
                    guard isValidIdentifier(trimmed) else {
                        throw ChatTemplateError.syntax("Invalid macro parameter: \(trimmed)")
                    }
                    params.append(ChatTemplateMacroParam(name: trimmed, defaultValue: nil))
                }
            }
        }

        let (body, terminator) = try parseNodes(until: ["endmacro"])
        guard terminator != nil else {
            throw ChatTemplateError.syntax("Missing endmacro")
        }

        return .macro(name: name, params: params, body: body)
    }

    private func splitMacroParams(_ value: String) -> [String] {
        var parts: [String] = []
        var current = ""
        var parenDepth = 0
        var quote: Character?

        for character in value {
            if let q = quote {
                current.append(character)
                if character == q {
                    quote = nil
                }
                continue
            }

            if character == "'" || character == "\"" {
                quote = character
                current.append(character)
                continue
            }

            if character == "(" {
                parenDepth += 1
                current.append(character)
                continue
            }

            if character == ")" {
                parenDepth -= 1
                current.append(character)
                continue
            }

            if character == "," && parenDepth == 0 {
                parts.append(current)
                current = ""
                continue
            }

            current.append(character)
        }

        if !current.isEmpty {
            parts.append(current)
        }

        return parts
    }

    private func readText() -> String {
        let start = index
        while index < characters.count, !matches("{{", at: index), !matches("{%", at: index) {
            index += 1
        }
        return String(characters[start..<index])
    }

    private func readTag(open: String, close: String) throws -> (content: String, trimLeft: Bool, trimRight: Bool) {
        index += open.count

        let trimLeft = index < characters.count && characters[index] == "-"
        if trimLeft {
            index += 1
        }

        let contentStart = index
        var quote: Character?
        var escaped = false

        while index < characters.count {
            let character = characters[index]

            if let q = quote {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == q {
                    quote = nil
                }
                index += 1
                continue
            }

            if character == "'" || character == "\"" {
                quote = character
                index += 1
                continue
            }

            if character == "-", matches(close, at: index + 1) {
                let content = String(characters[contentStart..<index])
                index += 1 + close.count
                return (content, trimLeft, true)
            }

            if matches(close, at: index) {
                let content = String(characters[contentStart..<index])
                index += close.count
                return (content, trimLeft, false)
            }

            index += 1
        }

        throw ChatTemplateError.syntax("Unterminated tag")
    }

    private func trimTrailingWhitespace(in nodes: inout [ChatTemplateNode]) {
        guard let lastNode = nodes.last else {
            return
        }

        guard case .text(let text) = lastNode else {
            return
        }

        let trimmed = trimTrailingWhitespace(from: text)
        nodes.removeLast()
        if !trimmed.isEmpty {
            nodes.append(.text(trimmed))
        }
    }

    private func trimLeadingWhitespace(from text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var start = 0
        while start < scalars.count, CharacterSet.whitespacesAndNewlines.contains(scalars[start]) {
            start += 1
        }
        guard start < scalars.count else {
            return ""
        }
        return String(String.UnicodeScalarView(scalars[start...]))
    }

    private func trimTrailingWhitespace(from text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var end = scalars.count
        while end > 0, CharacterSet.whitespacesAndNewlines.contains(scalars[end - 1]) {
            end -= 1
        }
        guard end > 0 else {
            return ""
        }
        return String(String.UnicodeScalarView(scalars[..<end]))
    }

    private func matches(_ value: String, at position: Int) -> Bool {
        let token = Array(value)
        guard position >= 0, position + token.count <= characters.count else {
            return false
        }

        for offset in 0..<token.count where characters[position + offset] != token[offset] {
            return false
        }

        return true
    }

    private func statementHead(for statement: String) -> String {
        statement.split(whereSeparator: { $0.isWhitespace }).first.map(String.init) ?? ""
    }

    private func splitTopLevel(_ value: String, separator: String) -> (String, String)? {
        let characters = Array(value)
        let separatorCharacters = Array(separator)
        guard characters.count >= separatorCharacters.count else {
            return nil
        }
        var quote: Character?
        var escaped = false
        var parenDepth = 0
        var bracketDepth = 0
        var position = 0

        while position <= characters.count - separatorCharacters.count {
            let character = characters[position]

            if let q = quote {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == q {
                    quote = nil
                }
                position += 1
                continue
            }

            if character == "'" || character == "\"" {
                quote = character
                position += 1
                continue
            }

            if character == "(" {
                parenDepth += 1
                position += 1
                continue
            }

            if character == ")" {
                parenDepth = max(0, parenDepth - 1)
                position += 1
                continue
            }

            if character == "[" {
                bracketDepth += 1
                position += 1
                continue
            }

            if character == "]" {
                bracketDepth = max(0, bracketDepth - 1)
                position += 1
                continue
            }

            if parenDepth == 0 && bracketDepth == 0 {
                let candidate = Array(characters[position..<(position + separatorCharacters.count)])
                if candidate == separatorCharacters {
                    let left = String(characters[..<position])
                    let right = String(characters[(position + separatorCharacters.count)...])
                    return (left, right)
                }
            }

            position += 1
        }

        return nil
    }

    private func isValidIdentifier(_ value: String) -> Bool {
        guard let first = value.first, first == "_" || first.isLetter else {
            return false
        }

        return value.dropFirst().allSatisfy { $0 == "_" || $0.isLetter || $0.isNumber }
    }
}

private enum ChatTemplateToken: Equatable {
    case identifier(String)
    case string(String)
    case integer(Int)
    case symbol(String)
    case end
}

private final class ChatTemplateTokenizer {
    private let characters: [Character]
    private var index = 0

    init(source: String) {
        self.characters = Array(source)
    }

    func tokenize() throws -> [ChatTemplateToken] {
        var tokens: [ChatTemplateToken] = []

        while true {
            skipWhitespace()
            guard index < characters.count else {
                tokens.append(.end)
                return tokens
            }

            let character = characters[index]

            if character == "'" || character == "\"" {
                tokens.append(.string(try readString(quote: character)))
                continue
            }

            if character.isNumber {
                tokens.append(.integer(readInteger()))
                continue
            }

            if character == "_" || character.isLetter {
                tokens.append(.identifier(readIdentifier()))
                continue
            }

            if match("==") {
                tokens.append(.symbol("=="))
                continue
            }

            if match("!=") {
                tokens.append(.symbol("!="))
                continue
            }

            let singleCharacterSymbols: Set<Character> = ["(", ")", "[", "]", ":", ",", ".", "+", "|", "-", "="]
            if singleCharacterSymbols.contains(character) {
                index += 1
                tokens.append(.symbol(String(character)))
                continue
            }

            throw ChatTemplateError.syntax("Unexpected token: \(character)")
        }
    }

    private func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace {
            index += 1
        }
    }

    private func readString(quote: Character) throws -> String {
        index += 1
        var value = ""

        while index < characters.count {
            let character = characters[index]
            index += 1

            if character == quote {
                return value
            }

            if character == "\\" {
                guard index < characters.count else {
                    throw ChatTemplateError.syntax("Unterminated escape sequence")
                }

                let escaped = characters[index]
                index += 1

                switch escaped {
                case "n":
                    value.append("\n")
                case "r":
                    value.append("\r")
                case "t":
                    value.append("\t")
                case "\\":
                    value.append("\\")
                case "'", "\"":
                    value.append(escaped)
                default:
                    value.append(escaped)
                }
                continue
            }

            value.append(character)
        }

        throw ChatTemplateError.syntax("Unterminated string literal")
    }

    private func readInteger() -> Int {
        let start = index
        while index < characters.count, characters[index].isNumber {
            index += 1
        }
        return Int(String(characters[start..<index])) ?? 0
    }

    private func readIdentifier() -> String {
        let start = index
        while index < characters.count, characters[index] == "_" || characters[index].isLetter || characters[index].isNumber {
            index += 1
        }
        return String(characters[start..<index])
    }

    private func match(_ value: String) -> Bool {
        let token = Array(value)
        guard index + token.count <= characters.count else {
            return false
        }

        for offset in 0..<token.count where characters[index + offset] != token[offset] {
            return false
        }

        index += token.count
        return true
    }
}

private final class ChatTemplateExpressionParser {
    private let tokens: [ChatTemplateToken]
    private var index = 0

    init(source: String) throws {
        self.tokens = try ChatTemplateTokenizer(source: source).tokenize()
    }

    func parse() throws -> ChatTemplateExpression {
        let expression = try parseOr()
        guard currentToken == .end else {
            throw ChatTemplateError.syntax("Unexpected token in expression")
        }
        return expression
    }

    private var currentToken: ChatTemplateToken {
        tokens[index]
    }

    private var nextToken: ChatTemplateToken {
        tokens[min(index + 1, tokens.count - 1)]
    }

    private func parseOr() throws -> ChatTemplateExpression {
        var expression = try parseAnd()
        while matchIdentifier("or") {
            expression = .binary(.or, expression, try parseAnd())
        }
        return expression
    }

    private func parseAnd() throws -> ChatTemplateExpression {
        var expression = try parseNot()
        while matchIdentifier("and") {
            expression = .binary(.and, expression, try parseNot())
        }
        return expression
    }

    private func parseNot() throws -> ChatTemplateExpression {
        if matchIdentifier("not") {
            return .unary(.not, try parseNot())
        }
        return try parseComparison()
    }

    private func parseComparison() throws -> ChatTemplateExpression {
        var expression = try parseAdditive()

        while true {
            if matchSymbol("==") {
                expression = .binary(.equals, expression, try parseAdditive())
                continue
            }

            if matchSymbol("!=") {
                expression = .binary(.notEquals, expression, try parseAdditive())
                continue
            }

            if matchIdentifier("in") {
                expression = .binary(.contains, expression, try parseAdditive())
                continue
            }

            if matchIdentifier("is") {
                let negated = matchIdentifier("not")
                if matchIdentifier("defined") {
                    expression = .isDefined(expression, negated: negated)
                } else if matchIdentifier("none") {
                    expression = .isTest(expression, "none", negated: negated)
                } else if matchIdentifier("undefined") {
                    expression = .isTest(expression, "undefined", negated: negated)
                } else if matchIdentifier("string") {
                    expression = .isTest(expression, "string", negated: negated)
                } else if matchIdentifier("true") {
                    expression = .isTest(expression, "true", negated: negated)
                } else if matchIdentifier("false") {
                    expression = .isTest(expression, "false", negated: negated)
                } else if matchIdentifier("mapping") {
                    expression = .isTest(expression, "mapping", negated: negated)
                } else if matchIdentifier("iterable") {
                    expression = .isTest(expression, "iterable", negated: negated)
                } else if matchIdentifier("sequence") {
                    expression = .isTest(expression, "sequence", negated: negated)
                } else if matchIdentifier("integer") || matchIdentifier("number") {
                    expression = .isTest(expression, "number", negated: negated)
                } else {
                    throw ChatTemplateError.syntax("Unknown test after 'is'")
                }
                continue
            }

            return expression
        }
    }

    private func parseAdditive() throws -> ChatTemplateExpression {
        var expression = try parsePostfix()
        while matchSymbol("+") {
            expression = .binary(.add, expression, try parsePostfix())
        }
        return expression
    }

    private func parsePostfix() throws -> ChatTemplateExpression {
        var expression = try parsePrimary()

        while true {
            if matchSymbol(".") {
                let name = try readIdentifier()
                expression = .member(expression, name)
                continue
            }

            if matchSymbol("[") {
                if matchSymbol(":") {
                    try expectSymbol("]")
                    expression = .slice(expression, start: nil)
                    continue
                }

                let start = try parseOr()
                if matchSymbol(":") {
                    try expectSymbol("]")
                    expression = .slice(expression, start: start)
                    continue
                }

                try expectSymbol("]")
                expression = .subscriptAccess(expression, start)
                continue
            }

            if matchSymbol("(") {
                expression = .call(expression, try parseCallArguments())
                continue
            }

            if matchSymbol("|") {
                expression = .filter(expression, try readIdentifier())
                continue
            }

            return expression
        }
    }

    private func parsePrimary() throws -> ChatTemplateExpression {
        switch currentToken {
        case .string(let value):
            index += 1
            return .literal(.string(value))
        case .integer(let value):
            index += 1
            return .literal(.int(value))
        case .identifier(let name):
            index += 1
            switch name {
            case "true":
                return .literal(.bool(true))
            case "false":
                return .literal(.bool(false))
            default:
                return .variable(name)
            }
        case .symbol("("):
            index += 1
            let expression = try parseOr()
            try expectSymbol(")")
            return expression
        case .symbol("-"):
            index += 1
            return .unary(.negate, try parsePrimary())
        default:
            throw ChatTemplateError.syntax("Unexpected token in expression")
        }
    }

    private func parseCallArguments() throws -> [ChatTemplateCallArgument] {
        var arguments: [ChatTemplateCallArgument] = []

        if matchSymbol(")") {
            return arguments
        }

        while true {
            if case .identifier(let name) = currentToken, nextToken == .symbol("=") {
                index += 1
                try expectSymbol("=")
                arguments.append(.named(name, try parseOr()))
            } else {
                arguments.append(.positional(try parseOr()))
            }

            if matchSymbol(")") {
                return arguments
            }

            try expectSymbol(",")
        }
    }

    private func matchIdentifier(_ value: String) -> Bool {
        guard case .identifier(let identifier) = currentToken, identifier == value else {
            return false
        }
        index += 1
        return true
    }

    private func matchSymbol(_ value: String) -> Bool {
        guard currentToken == .symbol(value) else {
            return false
        }
        index += 1
        return true
    }

    private func expectSymbol(_ value: String) throws {
        guard matchSymbol(value) else {
            throw ChatTemplateError.syntax("Expected '\(value)'")
        }
    }

    private func readIdentifier() throws -> String {
        guard case .identifier(let value) = currentToken else {
            throw ChatTemplateError.syntax("Expected identifier")
        }
        index += 1
        return value
    }
}

private final class ChatTemplateEvaluator {
    private var scopes: [[String: Any]]

    init(rootContext: [String: Any]) {
        self.scopes = [rootContext]
    }

    func render(_ nodes: [ChatTemplateNode]) throws -> String {
        var output = ""
        try render(nodes, into: &output)
        return output
    }

    private func render(_ nodes: [ChatTemplateNode], into output: inout String) throws {
        for node in nodes {
            switch node {
            case .text(let text):
                output.append(text)
            case .output(let expression):
                output.append(stringValue(try evaluate(expression)))
            case .forLoop(let variable, let iterable, let body, let elseBody):
                let sequence = sequenceValues(from: try evaluate(iterable))
                if sequence.isEmpty {
                    if let elseBody {
                        try render(elseBody, into: &output)
                    }
                    continue
                }

                for (index, item) in sequence.enumerated() {
                    pushScope([
                        variable: item,
                        "loop": [
                            "index0": index,
                            "first": index == 0,
                            "last": index == sequence.count - 1,
                            "length": sequence.count,
                        ],
                    ])
                    try render(body, into: &output)
                    popScope()
                }
            case .conditional(let clauses):
                for clause in clauses {
                    if let condition = clause.condition {
                        if truthy(try evaluate(condition)) {
                            try render(clause.body, into: &output)
                            break
                        }
                    } else {
                        try render(clause.body, into: &output)
                        break
                    }
                }
            case .setVariable(let name, let value):
                scopes[scopes.count - 1][name] = try evaluate(value)
            case .setAttribute(let target, let attribute, let value):
                guard let namespace = lookup(target) as? ChatTemplateNamespace else {
                    throw ChatTemplateError.runtime("Variable '\(target)' is not a namespace")
                }
                namespace.values[attribute] = try evaluate(value)
            case .macro(let name, let params, let body):
                scopes[scopes.count - 1][name] = ChatTemplateMacro(params: params, body: body)
            }
        }
    }

    private func evaluate(_ expression: ChatTemplateExpression) throws -> Any {
        switch expression {
        case .literal(let literal):
            switch literal {
            case .string(let value):
                return value
            case .int(let value):
                return value
            case .bool(let value):
                return value
            }
        case .variable(let name):
            return lookup(name)
        case .unary(let operation, let value):
            let resolved = try evaluate(value)
            switch operation {
            case .not:
                return !truthy(resolved)
            case .negate:
                guard let integer = intValue(resolved) else {
                    throw ChatTemplateError.runtime("Unary minus expects an integer")
                }
                return -integer
            }
        case .binary(let operation, let left, let right):
            let lhs = try evaluate(left)
            switch operation {
            case .and:
                if !truthy(lhs) {
                    return false
                }
                return truthy(try evaluate(right))
            case .or:
                if truthy(lhs) {
                    return true
                }
                return truthy(try evaluate(right))
            case .add:
                let rhs = try evaluate(right)
                if let leftInt = intValue(lhs), let rightInt = intValue(rhs), isInteger(lhs), isInteger(rhs) {
                    return leftInt + rightInt
                }
                return stringValue(lhs) + stringValue(rhs)
            case .equals:
                return valuesEqual(lhs, try evaluate(right))
            case .notEquals:
                return !valuesEqual(lhs, try evaluate(right))
            case .contains:
                return contains(value: lhs, in: try evaluate(right))
            }
        case .isDefined(let wrapped, let negated):
            let isDefined = !isUndefined(try evaluate(wrapped))
            return negated ? !isDefined : isDefined
        case .isTest(let wrapped, let testName, let negated):
            let value = try evaluate(wrapped)
            let result: Bool
            switch testName {
            case "none":
                result = isUndefined(value) || value is NSNull
            case "undefined":
                result = isUndefined(value)
            case "string":
                result = !isUndefined(value) && value is String
            case "true":
                result = (value as? Bool) == true
            case "false":
                result = (value as? Bool) == false
            case "mapping":
                result = !isUndefined(value) && value is [String: Any]
            case "iterable":
                result = !isUndefined(value) && (value is [Any] || value is [String: Any] || value is String)
            case "sequence":
                result = !isUndefined(value) && (value is [Any] || value is String)
            case "number":
                result = !isUndefined(value) && (value is Int || value is Double || value is Float)
            default:
                throw ChatTemplateError.runtime("Unknown test: \(testName)")
            }
            return negated ? !result : result
        case .member(let base, let name):
            return try member(named: name, on: try evaluate(base))
        case .subscriptAccess(let base, let index):
            return try subscriptValue(on: try evaluate(base), with: try evaluate(index))
        case .slice(let base, let start):
            let startValue = try start.map { try evaluate($0) }
            return try sliceValue(on: try evaluate(base), start: startValue)
        case .call(let callee, let arguments):
            return try call(callee: callee, arguments: arguments)
        case .filter(let base, let name):
            return try applyFilter(named: name, to: try evaluate(base))
        }
    }

    private func pushScope(_ scope: [String: Any]) {
        scopes.append(scope)
    }

    private func popScope() {
        _ = scopes.popLast()
    }

    private func lookup(_ name: String) -> Any {
        for scope in scopes.reversed() {
            if let value = scope[name] {
                return value
            }
        }
        return ChatTemplateUndefined.shared
    }

    private func truthy(_ value: Any) -> Bool {
        if isUndefined(value) {
            return false
        }
        if let bool = value as? Bool {
            return bool
        }
        if let integer = value as? Int {
            return integer != 0
        }
        if let string = value as? String {
            return !string.isEmpty
        }
        if let array = value as? [Any] {
            return !array.isEmpty
        }
        if let dictionary = value as? [String: Any] {
            return !dictionary.isEmpty
        }
        if let namespace = value as? ChatTemplateNamespace {
            return !namespace.values.isEmpty
        }
        return true
    }

    private func stringValue(_ value: Any) -> String {
        if isUndefined(value) {
            return ""
        }
        if let string = value as? String {
            return string
        }
        if let integer = value as? Int {
            return String(integer)
        }
        if let bool = value as? Bool {
            return bool ? "true" : "false"
        }
        return ""
    }

    private func intValue(_ value: Any) -> Int? {
        if let integer = value as? Int {
            return integer
        }
        if let string = value as? String {
            return Int(string)
        }
        return nil
    }

    private func isInteger(_ value: Any) -> Bool {
        value is Int
    }

    private func valuesEqual(_ left: Any, _ right: Any) -> Bool {
        if isUndefined(left) || isUndefined(right) {
            return isUndefined(left) && isUndefined(right)
        }
        if let leftString = left as? String, let rightString = right as? String {
            return leftString == rightString
        }
        if let leftInt = left as? Int, let rightInt = right as? Int {
            return leftInt == rightInt
        }
        if let leftBool = left as? Bool, let rightBool = right as? Bool {
            return leftBool == rightBool
        }
        return stringValue(left) == stringValue(right)
    }

    private func contains(value: Any, in container: Any) -> Bool {
        if let string = container as? String {
            return string.contains(stringValue(value))
        }
        if let array = container as? [Any] {
            return array.contains { valuesEqual($0, value) }
        }
        if let dictionary = container as? [String: Any] {
            return dictionary[stringValue(value)] != nil
        }
        if let namespace = container as? ChatTemplateNamespace {
            return namespace.values[stringValue(value)] != nil
        }
        return false
    }

    private func member(named name: String, on base: Any) throws -> Any {
        if isUndefined(base) {
            return ChatTemplateUndefined.shared
        }
        if let dictionary = base as? [String: Any] {
            return dictionary[name] ?? ChatTemplateUndefined.shared
        }
        if let namespace = base as? ChatTemplateNamespace {
            return namespace.values[name] ?? ChatTemplateUndefined.shared
        }
        throw ChatTemplateError.runtime("Unsupported attribute access: \(name)")
    }

    private func subscriptValue(on base: Any, with index: Any) throws -> Any {
        if isUndefined(base) {
            return ChatTemplateUndefined.shared
        }

        if let dictionary = base as? [String: Any] {
            return dictionary[stringValue(index)] ?? ChatTemplateUndefined.shared
        }

        if let namespace = base as? ChatTemplateNamespace {
            return namespace.values[stringValue(index)] ?? ChatTemplateUndefined.shared
        }

        if let array = base as? [Any] {
            guard let rawIndex = intValue(index) else {
                throw ChatTemplateError.runtime("List index must be an integer")
            }
            let resolvedIndex = rawIndex >= 0 ? rawIndex : array.count + rawIndex
            guard array.indices.contains(resolvedIndex) else {
                return ChatTemplateUndefined.shared
            }
            return array[resolvedIndex]
        }

        throw ChatTemplateError.runtime("Unsupported subscript access")
    }

    private func sliceValue(on base: Any, start: Any?) throws -> Any {
        if isUndefined(base) {
            return [Any]()
        }

        guard let array = base as? [Any] else {
            throw ChatTemplateError.runtime("Slicing is only supported for lists")
        }

        let startIndex: Int
        if let start {
            guard let rawIndex = intValue(start) else {
                throw ChatTemplateError.runtime("Slice index must be an integer")
            }
            startIndex = rawIndex >= 0 ? rawIndex : max(0, array.count + rawIndex)
        } else {
            startIndex = 0
        }

        let clampedIndex = min(max(0, startIndex), array.count)
        return Array(array[clampedIndex...])
    }

    private func call(callee: ChatTemplateExpression, arguments: [ChatTemplateCallArgument]) throws -> Any {
        switch callee {
        case .variable(let name):
            return try callBuiltin(named: name, arguments: arguments)
        case .member(let baseExpression, let methodName):
            let base = try evaluate(baseExpression)
            return try callMethod(named: methodName, on: base, arguments: arguments)
        default:
            throw ChatTemplateError.runtime("Unsupported call expression")
        }
    }

    private func callBuiltin(named name: String, arguments: [ChatTemplateCallArgument]) throws -> Any {
        // Check for user-defined macros first
        let resolved = lookup(name)
        if let macro = resolved as? ChatTemplateMacro {
            return try callMacro(macro, arguments: arguments)
        }

        switch name {
        case "raise_exception":
            let messageExpression = try requirePositionalArguments(arguments, expected: 1)[0]
            let message = try evaluate(messageExpression)
            throw ChatTemplateError.runtime(stringValue(message))
        case "namespace":
            var values: [String: Any] = [:]
            for argument in arguments {
                switch argument {
                case .named(let key, let value):
                    values[key] = try evaluate(value)
                case .positional:
                    throw ChatTemplateError.runtime("namespace() only accepts named arguments")
                }
            }
            return ChatTemplateNamespace(values: values)
        case "range":
            let values = try requirePositionalArguments(arguments, expected: nil).map { try evaluate($0) }
            let bounds: (Int, Int)
            if values.count == 1 {
                guard let end = intValue(values[0]) else {
                    throw ChatTemplateError.runtime("range() expects integer arguments")
                }
                bounds = (0, end)
            } else if values.count == 2 {
                guard let start = intValue(values[0]), let end = intValue(values[1]) else {
                    throw ChatTemplateError.runtime("range() expects integer arguments")
                }
                bounds = (start, end)
            } else {
                throw ChatTemplateError.runtime("range() expects one or two arguments")
            }
            return Array(bounds.0..<bounds.1).map { $0 as Any }
        default:
            throw ChatTemplateError.runtime("Unknown function: \(name)")
        }
    }

    private func callMacro(_ macro: ChatTemplateMacro, arguments: [ChatTemplateCallArgument]) throws -> Any {
        var scope: [String: Any] = [:]

        // Separate positional and named arguments
        var positionalArgs: [ChatTemplateExpression] = []
        var namedArgs: [String: ChatTemplateExpression] = [:]
        for argument in arguments {
            switch argument {
            case .positional(let expr):
                positionalArgs.append(expr)
            case .named(let name, let expr):
                namedArgs[name] = expr
            }
        }

        // Bind parameters
        for (paramIndex, param) in macro.params.enumerated() {
            if paramIndex < positionalArgs.count {
                scope[param.name] = try evaluate(positionalArgs[paramIndex])
            } else if let namedExpr = namedArgs[param.name] {
                scope[param.name] = try evaluate(namedExpr)
            } else if let defaultExpr = param.defaultValue {
                scope[param.name] = try evaluate(defaultExpr)
            } else {
                throw ChatTemplateError.runtime("Missing argument for macro parameter: \(param.name)")
            }
        }

        // Render body in new scope
        pushScope(scope)
        var output = ""
        try render(macro.body, into: &output)
        popScope()

        return output
    }

    private func callMethod(named name: String, on base: Any, arguments: [ChatTemplateCallArgument]) throws -> Any {
        _ = try requirePositionalArguments(arguments, expected: 0)
        let value = stringValue(base)

        switch name {
        case "strip":
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        case "title":
            return value.capitalized
        default:
            throw ChatTemplateError.runtime("Unknown method: \(name)")
        }
    }

    private func applyFilter(named name: String, to value: Any) throws -> Any {
        switch name {
        case "trim":
            return stringValue(value).trimmingCharacters(in: .whitespacesAndNewlines)
        case "length":
            if let string = value as? String {
                return string.count
            }
            if let array = value as? [Any] {
                return array.count
            }
            if let dictionary = value as? [String: Any] {
                return dictionary.count
            }
            if let namespace = value as? ChatTemplateNamespace {
                return namespace.values.count
            }
            return 0
        default:
            throw ChatTemplateError.runtime("Unknown filter: \(name)")
        }
    }

    private func requirePositionalArguments(
        _ arguments: [ChatTemplateCallArgument],
        expected: Int?
    ) throws -> [ChatTemplateExpression] {
        let positional = try arguments.map { argument -> ChatTemplateExpression in
            switch argument {
            case .positional(let expression):
                return expression
            case .named(let name, _):
                throw ChatTemplateError.runtime("Unexpected named argument: \(name)")
            }
        }

        if let expected, positional.count != expected {
            throw ChatTemplateError.runtime("Expected \(expected) arguments")
        }

        return positional
    }

    private func sequenceValues(from value: Any) -> [Any] {
        if let array = value as? [Any] {
            return array
        }
        return []
    }

    private func isUndefined(_ value: Any) -> Bool {
        value is ChatTemplateUndefined
    }
}
