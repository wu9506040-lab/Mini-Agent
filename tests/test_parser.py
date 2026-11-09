"""解析层测试：FC 提取 / 文本协议 XML / 接口可替换。"""
import asyncio

from app.agent.parser import BaseParser, FunctionCallingParser, ParsedOutput, TextParser
from app.llm.client import ToolCall

from conftest import make_llm_response


def test_fc_parser_extracts_tool_calls():
    resp = make_llm_response(
        content="我需要先计算",
        tool_calls=[ToolCall(id="c1", name="calculator",
                             arguments='{"expression": "1+1"}')],
    )
    out = FunctionCallingParser().parse(resp)
    assert out.is_final is False
    assert out.thought == "我需要先计算"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].parsed_arguments == {"expression": "1+1"}


def test_fc_parser_final_answer():
    out = FunctionCallingParser().parse(make_llm_response(content="答案是 42"))
    assert out.is_final is True
    assert out.tool_calls == []
    assert out.thought == "答案是 42"


def test_text_parser_extracts_xml_tool_call():
    content = '思考：需要计算。<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    out = TextParser().parse(make_llm_response(content=content))
    assert out.is_final is False
    assert "思考" in out.thought and "<tool_call>" not in out.thought
    assert out.tool_calls[0].name == "calculator"
    assert out.raw_message["tool_calls"][0]["function"]["name"] == "calculator"


def test_text_parser_plain_text_is_final():
    out = TextParser().parse(make_llm_response(content="直接回答"))
    assert out.is_final is True


def test_parser_interface_replaceable(make_runtime):
    class DummyParser(BaseParser):
        def parse(self, response):
            return ParsedOutput(thought="dummy", is_final=True)

    # 脚本里明明是工具调用，但 DummyParser 强制按最终回答处理——证明注入生效
    rt, _ = make_runtime([{"tool": ("calculator", {"expression": "1+1"})}],
                         parser=DummyParser())
    reply = asyncio.run(rt.run("s1", "算数"))
    assert reply.content == "dummy"
    assert reply.tool_calls == []
