"""Agent tools: web search, calculator, current time."""
import json
import math
import re
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Any


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取最新信息。用于查找知识库中没有的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持基本运算、三角函数、对数等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 2+3*4"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取当前日期和时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "len": len,
    "int": int, "float": float, "str": str,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "web_search":
        return _web_search(arguments.get("query", ""))
    elif name == "calculator":
        return _calculator(arguments.get("expression", ""))
    elif name == "current_time":
        return _current_time()
    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


def _web_search(query: str) -> str:
    if not query.strip():
        return json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)
    try:
        q = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "KnowledgeQA/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        results = []
        for r in data.get("RelatedTopics", [])[:5]:
            if "Text" in r:
                results.append(r["Text"])
        if results:
            return json.dumps({"results": results}, ensure_ascii=False)
        return json.dumps({"results": ["未找到相关结果"]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _calculator(expression: str) -> str:
    if not expression.strip():
        return json.dumps({"error": "表达式为空"}, ensure_ascii=False)
    safe = re.sub(r"[^0-9+\-*/().%^, ]", "", expression)
    if not safe:
        return json.dumps({"error": "无效表达式"}, ensure_ascii=False)
    try:
        result = eval(safe, {"__builtins__": {}}, SAFE_BUILTINS)
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _current_time() -> str:
    now = datetime.now()
    return json.dumps({
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][now.weekday()],
    }, ensure_ascii=False)
