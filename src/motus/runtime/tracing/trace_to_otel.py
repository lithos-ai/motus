"""
Convert tracer_state.json to OpenTelemetry format and generate visualization.

This script:
1. Reads tracer_state.json
2. Converts to OpenTelemetry spans format
3. Exports to Jaeger-compatible JSON
4. Generates an interactive HTML viewer

Usage:
    python trace_to_otel.py
    # Opens trace_viewer.html in browser
"""

import json
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

from ..types import MODEL_CALL, TOOL_CALL
from .pricing import TEXT_MODEL_PRICING


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for pricing lookup.

    Strips date suffixes (e.g., "claude-sonnet-4-20250514" -> "claude-sonnet-4")
    and handles common variations.
    """
    if not model_name:
        return model_name
    # Strip date suffixes (e.g., -20250514, -2025-05-14)
    normalized = model_name.split("-202")[0] if "-202" in model_name else model_name
    return normalized


def load_trace_data(path: str = "tracer_state.json") -> Dict[str, Any]:
    """Load the tracer state JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def convert_single_span_to_otel(
    task_id: str | int, meta: Dict[str, Any], trace_id: str
) -> Dict[str, Any]:
    """Convert a single task's metadata to an OpenTelemetry span dict.

    Args:
        task_id: The task identifier (string or int).
        meta: The task metadata dictionary.
        trace_id: The trace ID shared by all spans in this trace.

    Returns:
        A single OTel-format span dictionary.
    """
    # Times are already in microseconds since epoch
    start_time = meta.get("start_us", 0)
    end_time = meta.get("end_us", start_time)
    duration = end_time - start_time

    # Build attributes from metadata
    attributes = {
        "task.id": task_id,
        "task.func": meta.get("func", "unknown"),
    }

    # Add parent reference
    parent_span_id = None
    if meta.get("parent") is not None:
        parent_span_id = f"span-{meta['parent']}"

    # Add model metadata
    if meta.get("model_input_meta"):
        attributes["model.input.messages"] = len(meta["model_input_meta"])
    if meta.get("model_output_meta"):
        output = meta["model_output_meta"]
        if output.get("model"):
            attributes["model.name"] = output["model"]
        if output.get("usage"):
            usage = output["usage"]
            attributes["model.tokens.total"] = usage.get("total_tokens", 0)
            attributes["model.tokens.prompt"] = usage.get("prompt_tokens", 0)
            attributes["model.tokens.completion"] = usage.get("completion_tokens", 0)
            if usage.get("completion_tokens_details", {}).get("reasoning_tokens"):
                attributes["model.tokens.reasoning"] = usage[
                    "completion_tokens_details"
                ]["reasoning_tokens"]
            # Normalize model name for pricing lookup (strip date suffixes)
            pricing_model_name = normalize_model_name(meta.get("model_name", ""))
            if pricing_model_name in TEXT_MODEL_PRICING:
                pricing = TEXT_MODEL_PRICING[pricing_model_name]
                input_k = usage.get("prompt_tokens", 0) / 1000
                output_k = usage.get("completion_tokens", 0) / 1000
                cost = input_k * pricing.input_per_1k + output_k * pricing.output_per_1k
                attributes["model.cost_usd"] = round(cost, 5)

    # Add tool schema metadata
    if meta.get("tool_meta"):
        attributes["tools.available"] = len(meta["tool_meta"])
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in meta["tool_meta"]
            if t.get("function")
        ]
        if tool_names:
            attributes["tools.names"] = ", ".join(tool_names)

    # Add tool execution metadata
    if meta.get("tool_input_meta"):
        tool_input = meta["tool_input_meta"]
        if isinstance(tool_input, dict):
            # New format: {"name": "web_search", "arguments": {...}}
            if tool_input.get("name"):
                attributes["tool.name"] = tool_input["name"]

    # Determine span kind and display name
    span_kind = "INTERNAL"
    func_name = meta.get("func", "unknown")
    display_name = func_name  # Default to function name

    task_type = meta.get("task_type", "")
    if task_type == MODEL_CALL:
        span_kind = "CLIENT"
        # Use model name for display
        model_name = meta.get("model_name")
        if not model_name and meta.get("model_output_meta"):
            model_name = meta["model_output_meta"].get("model")
        if model_name:
            display_name = (
                model_name.split("-202")[0] if "-202" in model_name else model_name
            )
    elif task_type == TOOL_CALL:
        span_kind = "CLIENT"

    return {
        "traceId": trace_id,
        "spanId": f"span-{task_id}",
        "parentSpanId": parent_span_id,
        "operationName": display_name,
        "startTime": start_time,
        "duration": duration,
        "tags": attributes,
        "logs": [],
        "references": [],
        "kind": span_kind,
        "meta": meta,  # Keep original metadata for detailed view
    }


def convert_to_otel_spans(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert tracer format to OpenTelemetry span format."""
    trace_id = "trace-" + str(int(time.time()))
    return [
        convert_single_span_to_otel(task_id, meta, trace_id)
        for task_id, meta in trace_data.items()
    ]


def generate_html_viewer(
    spans: List[Dict[str, Any]],
    output_path: str = "trace_viewer.html",
    quiet: bool = False,
):
    """Generate an interactive HTML viewer for the trace."""
    # Get template directory path
    template_dir = Path(__file__).parent / "templates"

    # Load template files
    html_template = (template_dir / "trace_viewer.html").read_text()
    css_content = (template_dir / "trace_viewer.css").read_text()
    js_content = (template_dir / "trace_viewer.js").read_text()

    # Calculate timeline bounds
    min_time = min(s["startTime"] for s in spans) if spans else 0
    max_time = max(s["startTime"] + s["duration"] for s in spans) if spans else 0
    total_duration = max_time - min_time

    # Prepare data for embedding
    spans_json = json.dumps(spans, indent=2)

    # Replace placeholders in template
    html_content = html_template.replace("{{CSS_CONTENT}}", css_content)
    html_content = html_content.replace("{{JS_CONTENT}}", js_content)
    html_content = html_content.replace("{{SPANS_JSON}}", spans_json)
    html_content = html_content.replace("{{MIN_TIME}}", str(min_time))
    html_content = html_content.replace("{{TOTAL_DURATION}}", str(total_duration))

    with open(output_path, "w") as f:
        f.write(html_content)

    if not quiet:
        print(f"Generated trace viewer: {output_path}")
    return output_path


def export_jaeger_json(
    spans: List[Dict[str, Any]],
    output_path: str = "trace_jaeger.json",
    quiet: bool = False,
):
    """Export spans in Jaeger JSON format for compatibility with Jaeger UI."""

    if not spans:
        if not quiet:
            print("No spans to export")
        return

    trace_id = spans[0]["traceId"]

    # Convert to Jaeger format
    jaeger_spans = []
    for span in spans:
        jaeger_span = {
            "traceID": span["traceId"],
            "spanID": span["spanId"],
            "operationName": span["operationName"],
            "references": [],
            "startTime": span["startTime"],
            "duration": span["duration"],
            "tags": [
                {"key": k, "type": "string", "value": str(v)}
                for k, v in span["tags"].items()
            ],
            "logs": span.get("logs", []),
            "processID": "p1",
        }

        if span.get("parentSpanId"):
            jaeger_span["references"].append(
                {
                    "refType": "CHILD_OF",
                    "traceID": span["traceId"],
                    "spanID": span["parentSpanId"],
                }
            )

        jaeger_spans.append(jaeger_span)

    jaeger_data = {
        "data": [
            {
                "traceID": trace_id,
                "spans": jaeger_spans,
                "processes": {"p1": {"serviceName": "agent-runtime", "tags": []}},
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(jaeger_data, f, indent=2)

    if not quiet:
        print(f"✅ Exported Jaeger JSON: {output_path}")


def main():
    """Main execution function."""
    print("🔄 Loading trace data...")
    trace_data = load_trace_data()

    print("🔄 Converting to OpenTelemetry format...")
    spans = convert_to_otel_spans(trace_data)

    print(f"📊 Processed {len(spans)} spans")

    print("🔄 Generating HTML viewer...")
    html_path = generate_html_viewer(spans)

    print("🔄 Exporting Jaeger JSON...")
    export_jaeger_json(spans)

    print("\n✨ Done! Opening viewer in browser...")
    webbrowser.open(f"file://{Path(html_path).absolute()}")


if __name__ == "__main__":
    main()
