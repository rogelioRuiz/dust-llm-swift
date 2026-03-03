import XCTest
@testable import DustLlm

final class TestQwen35Template: XCTestCase {
    // Simplified Qwen 3.5 template (text-only, no vision/tool blocks)
    static let qwen35Template = """
        {%- set image_count = namespace(value=0) -%}
        {%- set video_count = namespace(value=0) -%}
        {%- macro render_content(content, do_vision_count, is_system_content=false) -%}
            {%- if content is string -%}
                {{- content -}}
            {%- elif content is none or content is undefined -%}
                {{- '' -}}
            {%- else -%}
                {{- raise_exception('Unexpected content type.') -}}
            {%- endif -%}
        {%- endmacro -%}
        {%- if not messages -%}
            {{- raise_exception('No messages provided.') -}}
        {%- endif -%}
        {%- if messages[0].role == 'system' -%}
            {%- set content = render_content(messages[0].content, false, true)|trim -%}
            {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' -}}
        {%- endif -%}
        {%- for message in messages -%}
            {%- set content = render_content(message.content, true)|trim -%}
            {%- if message.role == 'user' -%}
                {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' -}}
            {%- elif message.role == 'assistant' -%}
                {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' -}}
            {%- endif -%}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- '<|im_start|>assistant\\n' -}}
            {%- if enable_thinking is defined and enable_thinking is true -%}
                {{- '<think>\\n' -}}
            {%- else -%}
                {{- '<think>\\n\\n</think>\\n\\n' -}}
            {%- endif -%}
        {%- endif -%}
        """

    func testQwen35TemplateMultiTurn() throws {
        let engine = ChatTemplateEngine(templateString: Self.qwen35Template)
        let messages = [
            ChatMessage(role: "user", content: "Hello from the chat test"),
            ChatMessage(role: "assistant", content: "Hello! How can I assist you today?"),
            ChatMessage(role: "user", content: "What is 2+2?"),
        ]
        let output = try engine.apply(messages: messages, addGenerationPrompt: true)

        XCTAssertTrue(output.contains("<|im_start|>user\nHello from the chat test<|im_end|>"), "Missing first user turn")
        XCTAssertTrue(output.contains("<|im_start|>assistant\nHello! How can I assist you today?<|im_end|>"), "Missing assistant turn")
        XCTAssertTrue(output.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"), "Missing second user turn")
        XCTAssertTrue(output.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "Missing generation prompt with think tags")
    }
}
