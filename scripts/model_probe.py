#!/usr/bin/env python3
"""Model Response Probe — discover how models format tool calls.

Sends a series of prompts to the active model (from config), collects raw
responses, runs them through the tool parser, and generates a report of
patterns, failures, and edge cases.

Usage:
    # Run all probes against the active model profile
    uv run python scripts/model_probe.py

    # Run against a specific profile
    uv run python scripts/model_probe.py --profile ollama

    # Run only specific probe categories
    uv run python scripts/model_probe.py --category basic

    # Save raw responses for later analysis
    uv run python scripts/model_probe.py --save-raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supercoder.config import Config
from supercoder.llm.openai_client import OpenAIClient
from supercoder.llm.base import Message
from supercoder.agent.prompts import build_system_prompt
from supercoder.agent.tool_parser import ToolCallParser, _safe_json_loads, _repair_json
from supercoder.tools import ALL_TOOLS


# ─── Probe definitions ───────────────────────────────────────────────────

@dataclass
class Probe:
    """A single test prompt to send to the model."""
    id: str
    category: str
    description: str
    prompt: str
    # What we expect in the response
    expect_tool_call: bool = True
    expect_tool_name: str | None = None
    expect_multiple_tools: bool = False
    # Edge case flags
    tests_json_escaping: bool = False
    tests_multiline: bool = False
    tests_unicode: bool = False
    tests_multi_tool: bool = False


PROBES = [
    # ── Basic tool calls ──
    Probe(
        id="basic_read",
        category="basic",
        description="Simple file read",
        prompt="Read the file main.py",
        expect_tool_name="file-read",
    ),
    Probe(
        id="basic_structure",
        category="basic",
        description="Project structure",
        prompt="Show me the project structure",
        expect_tool_name="project-structure",
    ),
    Probe(
        id="basic_search",
        category="basic",
        description="Code search",
        prompt="Search for the word 'def' in all Python files",
        expect_tool_name="code-search",
    ),
    Probe(
        id="basic_command",
        category="basic",
        description="Simple command execution",
        prompt="Run the command: echo hello",
        expect_tool_name="command-exec",
    ),

    # ── JSON escaping edge cases ──
    Probe(
        id="escape_quotes",
        category="escaping",
        description="Code with quotes in arguments",
        prompt='Create a file called test.py with this content: print("Hello, World!")',
        expect_tool_name="code-edit",
        tests_json_escaping=True,
    ),
    Probe(
        id="escape_backslash",
        category="escaping",
        description="Code with backslashes",
        prompt=r'Create a file test.py containing: path = "C:\\Users\\test"',
        expect_tool_name="code-edit",
        tests_json_escaping=True,
    ),
    Probe(
        id="escape_newlines",
        category="escaping",
        description="Multi-line code in arguments",
        prompt=(
            "Replace the content of main.py with a function that has 3 lines: "
            "def hello():\\n    name = 'world'\\n    print(f'Hello {name}')"
        ),
        expect_tool_name="code-edit",
        tests_json_escaping=True,
        tests_multiline=True,
    ),
    Probe(
        id="escape_nested_json",
        category="escaping",
        description="Code containing JSON strings",
        prompt=(
            'Create a file config.py with: data = {"key": "value", "list": [1, 2, 3]}'
        ),
        expect_tool_name="code-edit",
        tests_json_escaping=True,
    ),

    # ── Multi-line / large code ──
    Probe(
        id="multiline_function",
        category="multiline",
        description="Create a function with multiple lines",
        prompt=(
            "Replace the content of main.py with a complete Python function "
            "that calculates factorial recursively, with error handling and docstring. "
            "About 15-20 lines."
        ),
        expect_tool_name="code-edit",
        tests_multiline=True,
    ),
    Probe(
        id="multiline_class",
        category="multiline",
        description="Create a class with methods",
        prompt=(
            "Replace main.py content with a Calculator class that has methods "
            "add, subtract, multiply, divide (with zero division handling). "
            "Include __init__ and __repr__."
        ),
        expect_tool_name="code-edit",
        tests_multiline=True,
    ),

    # ── Unicode ──
    Probe(
        id="unicode_content",
        category="unicode",
        description="Code with unicode characters",
        prompt=(
            "Create file greeting.py with: "
            'print("Привет мир! 你好世界! مرحبا")'
        ),
        expect_tool_name="code-edit",
        tests_unicode=True,
    ),

    # ── Multiple tool calls ──
    Probe(
        id="multi_read_edit",
        category="multi",
        description="Read then edit (should produce 2 tool calls or sequential)",
        prompt="Read main.py, then add a comment '# modified' at the top",
        expect_multiple_tools=True,
        tests_multi_tool=True,
    ),

    # ── No tool needed ──
    Probe(
        id="no_tool_question",
        category="no_tool",
        description="Question that doesn't need tools",
        prompt="What is the difference between a list and a tuple in Python?",
        expect_tool_call=False,
    ),

    # ── Error recovery ──
    Probe(
        id="recovery_retry",
        category="recovery",
        description="Model response after malformed JSON feedback",
        prompt="(This probe is handled specially — tests retry behavior)",
        expect_tool_call=True,
        expect_tool_name="file-read",
    ),
]


# ─── Analysis result ─────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of running a single probe."""
    probe: Probe
    raw_response: str = ""
    response_time_ms: int = 0
    reasoning_text: str = ""

    # Parse results
    tool_calls_found: int = 0
    tool_names: list[str] = field(default_factory=list)
    parse_format: str = ""  # which parser matched
    json_valid: bool = True
    json_needed_repair: bool = False

    # Detected patterns
    has_text_before_tool: bool = False
    has_text_after_tool: bool = False
    has_reasoning: bool = False
    tag_format_used: str = ""  # e.g. "<@TOOL>", "to=tool:", "```json"

    # Failures
    error: str = ""
    raw_json_error: str = ""  # JSON parse error message if applicable


# ─── Runner ───────────────────────────────────────────────────────────────

def build_system_message(tool_calling_type: str) -> str:
    """Build system prompt using real tools."""
    return build_system_prompt(
        ALL_TOOLS,
        tool_calling_type=tool_calling_type,
    )


def detect_tag_format(text: str) -> str:
    """Detect which tool call tag format the model used."""
    if "<@TOOL>" in text:
        return "<@TOOL>...</@TOOL>"
    if "to=tool:" in text.lower():
        return "to=tool:name {...}"
    if "```json" in text:
        return "```json ... ```"
    if "<function_call" in text:
        return "<function_call>...</function_call>"
    if "<tool_call>" in text:
        return "<tool_call>...</tool_call>"
    return "(none detected)"


def analyze_json_quality(text: str, tool_calling_type: str) -> tuple[bool, bool, str]:
    """Check JSON quality within tool call tags.

    Returns: (valid, needed_repair, error_message)
    """
    # Extract JSON based on format
    patterns = {
        "supercoder": (r"<@TOOL>(.*?)</@TOOL>", re.DOTALL),
        "json_block": (r"```json\s*(.*?)```", re.DOTALL),
        "xml_function": (r"<function_call[^>]*>(.*?)</function_call>", re.DOTALL),
        "glm_tool_call": (r"<tool_call>(.*?)</tool_call>", re.DOTALL),
    }

    pat_info = patterns.get(tool_calling_type)
    if not pat_info:
        return True, False, ""

    pattern, flags = pat_info
    match = re.search(pattern, text, flags)
    if not match:
        return True, False, ""

    json_text = match.group(1).strip()

    # Test 1: strict JSON
    try:
        json.loads(json_text)
        return True, False, ""
    except json.JSONDecodeError as e1:
        strict_error = str(e1)

    # Test 2: strict=False
    try:
        json.loads(json_text, strict=False)
        return True, True, f"strict failed ({strict_error}), strict=False OK"
    except json.JSONDecodeError:
        pass

    # Test 3: repair
    try:
        json.loads(_repair_json(json_text))
        return True, True, f"strict failed ({strict_error}), repair fixed it"
    except json.JSONDecodeError as e3:
        return False, True, f"unfixable: {e3}"


def run_probe(
    client: OpenAIClient,
    system_msg: str,
    probe: Probe,
    tool_calling_type: str,
    parser: ToolCallParser,
) -> ProbeResult:
    """Run a single probe and analyze the result."""
    result = ProbeResult(probe=probe)

    messages = [
        Message("system", system_msg),
        Message("user", probe.prompt),
    ]

    # Special handling for recovery probe
    if probe.id == "recovery_retry":
        messages = [
            Message("system", system_msg),
            Message("user", "Read the file main.py"),
            Message("assistant", '<@TOOL>{"name": "file-read", "arguments": BROKEN JSON}</@TOOL>'),
            Message("user", (
                "<@TOOL_RESULT>ERROR: Your tool call could not be parsed — the JSON "
                "was malformed. Please retry with properly escaped JSON.</@TOOL_RESULT>"
            )),
        ]

    try:
        # Use non-streaming for simplicity
        t0 = time.time()
        response = ""
        reasoning = ""

        for chunk in client.chat_stream(messages):
            if not chunk.is_done:
                response += chunk.content
                reasoning += chunk.reasoning
            
        result.response_time_ms = int((time.time() - t0) * 1000)
        result.raw_response = response
        result.reasoning_text = reasoning
        result.has_reasoning = bool(reasoning.strip())

    except Exception as e:
        result.error = f"API error: {e}"
        return result

    # ── Analyze response ──

    # Detect tag format
    result.tag_format_used = detect_tag_format(response)

    # Check for text around tool calls
    tag_markers = ["<@TOOL>", "to=tool:", "<function_call", "<tool_call>", "```json"]
    for marker in tag_markers:
        idx = response.find(marker)
        if idx >= 0:
            before = response[:idx].strip()
            result.has_text_before_tool = bool(before)
            break

    # Parse tool calls
    tool_calls = parser.parse_all(response)
    result.tool_calls_found = len(tool_calls)
    result.tool_names = [tc.name for tc in tool_calls]
    if tool_calls:
        result.parse_format = tool_calls[0].format_name

    # Check JSON quality
    json_valid, needed_repair, json_error = analyze_json_quality(
        response, tool_calling_type
    )
    result.json_valid = json_valid
    result.json_needed_repair = needed_repair
    result.raw_json_error = json_error

    return result


# ─── Report ───────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"


def print_result(r: ProbeResult) -> None:
    """Print a single probe result."""
    p = r.probe

    # Status icon
    if r.error:
        icon = FAIL
    elif not r.json_valid:
        icon = FAIL
    elif p.expect_tool_call and r.tool_calls_found == 0:
        icon = FAIL
    elif not p.expect_tool_call and r.tool_calls_found > 0:
        icon = WARN
    elif p.expect_tool_name and p.expect_tool_name not in r.tool_names:
        icon = WARN
    elif r.json_needed_repair:
        icon = WARN
    else:
        icon = PASS

    print(f"\n{icon}  [{p.id}] {p.description}")
    print(f"   Response: {len(r.raw_response)} chars, {r.response_time_ms}ms")

    if r.error:
        print(f"   ERROR: {r.error}")
        return

    # Tool calls
    if r.tool_calls_found:
        names = ", ".join(r.tool_names)
        print(f"   Tool calls: {r.tool_calls_found} → [{names}] via {r.parse_format}")
    else:
        expected = "expected" if p.expect_tool_call else "as expected"
        print(f"   Tool calls: 0 ({expected})")

    # Tag format
    print(f"   Tag format: {r.tag_format_used}")

    # JSON quality
    if r.raw_json_error:
        print(f"   JSON: {r.raw_json_error}")

    # Patterns
    patterns = []
    if r.has_text_before_tool:
        patterns.append("text-before-tool")
    if r.has_text_after_tool:
        patterns.append("text-after-tool")
    if r.has_reasoning:
        patterns.append("has-reasoning")
    if patterns:
        print(f"   Patterns: {', '.join(patterns)}")


def print_summary(results: list[ProbeResult], model: str, tool_type: str) -> None:
    """Print aggregate summary."""
    total = len(results)
    passed = sum(1 for r in results if not r.error and r.json_valid and
                 (not r.probe.expect_tool_call or r.tool_calls_found > 0))
    failed = total - passed
    repairs = sum(1 for r in results if r.json_needed_repair)
    avg_time = sum(r.response_time_ms for r in results) // max(total, 1)

    # Detect most common format
    formats = [r.tag_format_used for r in results if r.tag_format_used != "(none detected)"]
    common_format = max(set(formats), key=formats.count) if formats else "unknown"

    print("\n" + "=" * 70)
    print(f"  MODEL PROBE REPORT — {model}")
    print(f"  tool_calling_type: {tool_type}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n  Total probes:     {total}")
    print(f"  Passed:           {passed} {PASS}")
    print(f"  Failed:           {failed} {FAIL}")
    print(f"  Needed repair:    {repairs} {WARN}")
    print(f"  Avg response:     {avg_time}ms")
    print(f"  Tag format used:  {common_format}")

    # Detailed pattern analysis
    text_before = sum(1 for r in results if r.has_text_before_tool)
    has_reasoning = sum(1 for r in results if r.has_reasoning)

    print(f"\n  Patterns:")
    print(f"    Text before tool call:  {text_before}/{total}")
    print(f"    Uses reasoning tokens:  {has_reasoning}/{total}")

    # JSON issues breakdown
    if repairs or failed:
        print(f"\n  JSON Issues:")
        for r in results:
            if r.raw_json_error:
                print(f"    [{r.probe.id}] {r.raw_json_error}")

    # Failed probes
    if failed:
        print(f"\n  Failed Probes:")
        for r in results:
            if r.error or not r.json_valid or (r.probe.expect_tool_call and r.tool_calls_found == 0):
                reason = r.error or r.raw_json_error or "no tool call found"
                print(f"    [{r.probe.id}] {reason}")

    print("\n" + "=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Probe model tool call behavior")
    ap.add_argument("--profile", help="Model profile name from config")
    ap.add_argument("--category", help="Run only probes in this category")
    ap.add_argument("--probe", help="Run a single probe by ID")
    ap.add_argument("--save-raw", action="store_true", help="Save raw responses to JSONL")
    ap.add_argument("--output", default="probe_results.jsonl", help="Output file for --save-raw")
    args = ap.parse_args()

    # Load config
    config = Config.load()

    if args.profile:
        if not config.switch_to_model(args.profile):
            print(f"❌ Unknown profile: {args.profile}")
            available = ", ".join(config.get_available_models())
            print(f"   Available: {available}")
            return 1

    errors = config.validate()
    if errors:
        for e in errors:
            print(f"❌ Config error: {e}")
        return 1

    # Get tool calling type
    profile = config.get_model_profile(config.current_profile_name)
    tool_type = profile.tool_calling_type if profile else "supercoder"

    print(f"🔬 Model Probe — {config.model}")
    print(f"   Endpoint: {config.base_url}")
    print(f"   Tool calling type: {tool_type}")
    print(f"   Temperature: {config.temperature}")

    # Initialize
    client = OpenAIClient(config)
    system_msg = build_system_message(tool_type)
    parser = ToolCallParser(tool_type)

    # Filter probes
    probes = PROBES
    if args.category:
        probes = [p for p in probes if p.category == args.category]
    if args.probe:
        probes = [p for p in probes if p.id == args.probe]

    if not probes:
        print("❌ No probes matched the filter")
        return 1

    print(f"\n   Running {len(probes)} probes...\n")
    print("─" * 70)

    # Run probes
    results: list[ProbeResult] = []
    raw_outputs: list[dict] = []

    for i, probe in enumerate(probes, 1):
        print(f"\n[{i}/{len(probes)}] Sending: {probe.description}...", end="", flush=True)

        result = run_probe(client, system_msg, probe, tool_type, parser)
        results.append(result)

        # Quick status
        if result.error:
            print(f" {FAIL} error")
        elif result.tool_calls_found > 0:
            print(f" {PASS} {result.tool_calls_found} tool(s), {result.response_time_ms}ms")
        elif not probe.expect_tool_call:
            print(f" {PASS} no tool (expected), {result.response_time_ms}ms")
        else:
            print(f" {FAIL} no tool call parsed!")

        # Save raw
        if args.save_raw:
            raw_outputs.append({
                "probe_id": probe.id,
                "category": probe.category,
                "prompt": probe.prompt,
                "response": result.raw_response,
                "reasoning": result.reasoning_text,
                "tool_calls_found": result.tool_calls_found,
                "tool_names": result.tool_names,
                "parse_format": result.parse_format,
                "json_valid": result.json_valid,
                "json_needed_repair": result.json_needed_repair,
                "json_error": result.raw_json_error,
                "tag_format": result.tag_format_used,
                "response_time_ms": result.response_time_ms,
                "timestamp": datetime.now().isoformat(),
            })

    # Print detailed results
    print("\n" + "─" * 70)
    print("\nDETAILED RESULTS")
    print("─" * 70)
    for r in results:
        print_result(r)

    # Print summary
    print_summary(results, config.model, tool_type)

    # Save raw responses
    if args.save_raw and raw_outputs:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            for entry in raw_outputs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n📄 Raw responses saved to {output_path}")

    return 0 if all(
        not r.error and r.json_valid and (not r.probe.expect_tool_call or r.tool_calls_found > 0)
        for r in results
    ) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
