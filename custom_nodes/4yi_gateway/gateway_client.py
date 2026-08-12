"""Pure request/response logic for the 4yi Gateway custom nodes.

This module must stay free of ComfyUI / torch imports so it can be unit-tested
on a bare Python environment. All HTTP and tensor work lives in nodes.py.

Env contract (injected per-install by the 4yi App Platform):
  IMAGE_API_BASE / OPENAI_API_BASE  -> "<platform-origin>/api/v1"
  IMAGE_API_KEY  / OPENAI_API_KEY   -> per-install gateway token

No model env is injected: the caller's entitled models are discovered at
runtime from the gateway's /models endpoint (see parse_model_list).
"""

import json
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

BASE_ENV_CANDIDATES = ("IMAGE_API_BASE", "OPENAI_API_BASE")
KEY_ENV_CANDIDATES = ("IMAGE_API_KEY", "OPENAI_API_KEY")

IMAGE_N_MIN, IMAGE_N_MAX = 1, 10
VIDEO_DURATION_MIN, VIDEO_DURATION_MAX = 1, 120


class GatewayError(Exception):
    """Raised for configuration or gateway-reported failures."""


def _first_env(env: Mapping[str, str], names: Tuple[str, ...]) -> str:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_gateway_config(
    env: Mapping[str, str],
    override_base: str = "",
    override_key: str = "",
) -> Tuple[str, str]:
    base = (override_base or "").strip() or _first_env(env, BASE_ENV_CANDIDATES)
    key = (override_key or "").strip() or _first_env(env, KEY_ENV_CANDIDATES)
    if not base:
        raise GatewayError(
            "4yi gateway base URL not configured: set IMAGE_API_BASE (or OPENAI_API_BASE), "
            "or fill the node's api_base field"
        )
    if not key:
        raise GatewayError(
            "4yi gateway API key not configured: set IMAGE_API_KEY (or OPENAI_API_KEY), "
            "or fill the node's api_key field"
        )
    return base.rstrip("/"), key


# ── images ───────────────────────────────────────────────────────────────────

def build_image_payload(model: str, prompt: str, n: int, size: str) -> Dict[str, Any]:
    if not prompt.strip():
        raise GatewayError("prompt must not be empty")
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": min(max(int(n), IMAGE_N_MIN), IMAGE_N_MAX),
        "response_format": "b64_json",
    }
    if size and size != "auto":
        payload["size"] = size
    return payload


def parse_model_list(body: Mapping[str, Any], model_type: str) -> List[str]:
    """Ids of models of `model_type` from an OpenAI-style /models response.

    Accepts both `type`/`model_type` and `id`/`name` spellings so it works
    against the 4yi gateway (which returns {id, type}) without pinning one shape.
    """
    ids: List[str] = []
    for item in body.get("data") or []:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type") or item.get("model_type")
        model_id = item.get("id") or item.get("name")
        if item_type == model_type and model_id:
            ids.append(str(model_id))
    return ids


def parse_image_entries(body: Mapping[str, Any]) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for item in body.get("data") or []:
        if isinstance(item, Mapping):
            if item.get("b64_json"):
                entries.append(("b64", str(item["b64_json"])))
            elif item.get("url"):
                entries.append(("url", str(item["url"])))
    if not entries:
        raise GatewayError("gateway returned no image data")
    return entries


# ── video ────────────────────────────────────────────────────────────────────

def build_video_payload(
    model: str,
    prompt: str,
    duration_seconds: int,
    resolution: str,
    image_url: str = "",
    extra_body_json: str = "",
) -> Dict[str, Any]:
    if not prompt.strip():
        raise GatewayError("prompt must not be empty")
    duration = int(duration_seconds)
    if duration < VIDEO_DURATION_MIN or duration > VIDEO_DURATION_MAX:
        raise GatewayError(
            f"duration_seconds must be between {VIDEO_DURATION_MIN} and {VIDEO_DURATION_MAX}"
        )

    extra_body: Dict[str, Any] = {}
    if image_url.strip():
        extra_body["first_frame"] = image_url.strip()
    if extra_body_json.strip():
        try:
            parsed = json.loads(extra_body_json)
        except json.JSONDecodeError as error:
            raise GatewayError(f"extra_body is not valid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise GatewayError("extra_body must be a JSON object")
        extra_body.update(parsed)

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "duration_seconds": duration,
        "mode": "image_to_video" if image_url.strip() else "text_to_video",
    }
    if resolution.strip():
        payload["resolution"] = resolution.strip()
    if extra_body:
        payload["extra_body"] = extra_body
    return payload


def interpret_video_poll(body: Mapping[str, Any]) -> Optional[str]:
    """None while running; the artifact URL once completed; raises on failure."""
    status = body.get("status")
    if status == "failed":
        raise GatewayError(str(body.get("failure_message") or "video generation failed"))
    if status == "completed":
        url = str(body.get("video_url") or "").strip()
        if not url:
            raise GatewayError("gateway reported completion without a video_url")
        return url
    return None


async def poll_video_until_complete(
    fetch: Callable[[], Awaitable[Mapping[str, Any]]],
    sleep: Callable[[float], Awaitable[None]],
    interval_seconds: float,
    max_attempts: int,
) -> str:
    for attempt in range(max_attempts):
        url = interpret_video_poll(await fetch())
        if url is not None:
            return url
        if attempt < max_attempts - 1:
            await sleep(interval_seconds)
    raise GatewayError(
        f"video generation timed out after {max_attempts} polls ({interval_seconds}s interval)"
    )
