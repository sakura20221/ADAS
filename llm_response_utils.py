import ast
import json
import os
import re

PRINT_EMPTY_TRACE = os.environ.get("ADAS_PRINT_EMPTY_TRACE", "1") == "1"
_LAST_RAW_CONTENT = None
_LAST_RESPONSE_METADATA = None
NON_EMPTY_OUTPUT_FIELDS = {
    "answer",
    "choice",
    "code",
    "correct",
    "score",
    "name",
}


class EmptyLLMResponseError(RuntimeError):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _parse_marker_object(text: str) -> dict:
    marker = re.compile(r"""(['"])(?P<key>[A-Za-z_][A-Za-z0-9_]*?)\1\s*:\s*""")
    matches = list(marker.finditer(text))
    if not matches:
        return {}

    result = {}
    for index, match in enumerate(matches):
        key = match.group("key")
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_value = text[value_start:value_end].strip()
        raw_value = raw_value.rstrip(",").strip()
        raw_value = raw_value.strip("{}").strip()
        raw_value = _strip_wrapping(raw_value)
        result[key] = raw_value
    return result


def parse_llm_content(content):
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    text = _strip_code_fences(content)
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    parsed = _parse_marker_object(text)
    if parsed:
        return parsed

    answer_match = re.search(r"""['"]answer['"]\s*:\s*['"]?([ABCD])['"]?""", text)
    if answer_match:
        return {"answer": answer_match.group(1)}

    return {}


def extract_choice(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else ""


def extract_content(value):
    if isinstance(value, list) and value:
        candidate = value[1] if len(value) > 1 else value[0]
        return getattr(candidate, "content", candidate)
    return getattr(value, "content", value)


def normalize_response_fields(response_json, output_fields, raw_content=None):
    if not isinstance(response_json, dict):
        response_json = {}

    if "answer" in output_fields and "answer" not in response_json:
        for value in response_json.values():
            inferred_answer = extract_choice(value)
            if inferred_answer:
                response_json["answer"] = inferred_answer
                break
        if not response_json.get("answer"):
            response_json["answer"] = extract_choice(raw_content)

    missing_fields = []
    for key in output_fields:
        if key not in response_json:
            missing_fields.append(key)
        elif _requires_non_empty_value(key) and not str(response_json.get(key, "")).strip():
            missing_fields.append(key)
    return response_json, missing_fields


def _requires_non_empty_value(key):
    lowered = key.lower()
    return lowered in NON_EMPTY_OUTPUT_FIELDS or lowered.endswith("_answer")


def fill_missing_response_fields(response_json, output_fields):
    if not isinstance(response_json, dict):
        response_json = {}

    for key in output_fields:
        if key not in response_json and len(response_json) < len(output_fields):
            response_json[key] = extract_choice(response_json.get(key, "")) if "answer" in key else ""

    for key in list(response_json.keys()):
        if len(response_json) > len(output_fields) and key not in output_fields:
            del response_json[key]
    return response_json


def set_last_raw_content(content):
    global _LAST_RAW_CONTENT
    _LAST_RAW_CONTENT = content


def get_last_raw_content():
    return _LAST_RAW_CONTENT


def set_last_response_metadata(response):
    global _LAST_RESPONSE_METADATA
    if response is None:
        _LAST_RESPONSE_METADATA = None
        return

    choice = response.choices[0] if getattr(response, "choices", None) else None
    usage = getattr(response, "usage", None)
    _LAST_RESPONSE_METADATA = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else str(usage),
    }


def get_last_response_metadata():
    return _LAST_RESPONSE_METADATA


def raise_if_empty_response(content):
    if content is None or not str(content).strip():
        raise EmptyLLMResponseError("LLM returned empty content")


def trace_llm_failure(agent, system_prompt, prompt, response_json, raw_content=None, error=None):
    if not PRINT_EMPTY_TRACE:
        return
    print("=== LLM RESPONSE FAILURE TRACE ===")
    print(f"agent: {agent}")
    print(f"system_prompt: {system_prompt}")
    print(f"prompt: {prompt}")
    if raw_content is not None:
        print(f"raw_content: {raw_content!r}")
    if _LAST_RESPONSE_METADATA is not None:
        print(f"response_metadata: {_LAST_RESPONSE_METADATA!r}")
    print(f"parsed_response: {response_json!r}")
    if error is not None:
        print(f"error: {error}")
    print("=== END LLM RESPONSE FAILURE TRACE ===")
