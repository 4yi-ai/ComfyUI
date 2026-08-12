"""ComfyUI nodes that generate images/videos through the 4yi Gateway.

The gateway speaks the OpenAI images API (`POST {base}/images/generations`,
synchronous) and an async video API (`POST {base}/videos/generations` ->
202 {id}, then `GET {base}/videos/generations/{id}` until completed). Both
require a Bearer token; the App Platform injects only the gateway base URL and
per-install key (IMAGE_API_BASE / IMAGE_API_KEY). No model is chosen at install
time: each node fetches the caller's entitled models from `GET {base}/models`
and offers them as a dropdown, so every model in the org's plan is available.
"""

import asyncio
import base64
import json
import os
import urllib.request
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import torch

from comfy_api.latest import InputImpl
from comfy_api_nodes.util.conversions import bytesio_to_image_tensor

try:
    from .gateway_client import (
        GatewayError,
        build_image_payload,
        build_video_payload,
        parse_image_entries,
        parse_model_list,
        poll_video_until_complete,
        resolve_gateway_config,
    )
except ImportError:  # pragma: no cover - direct script/test import
    from gateway_client import (
        GatewayError,
        build_image_payload,
        build_video_payload,
        parse_image_entries,
        parse_model_list,
        poll_video_until_complete,
        resolve_gateway_config,
    )

REQUEST_TIMEOUT_SECONDS = 300
DOWNLOAD_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 5
MODELS_TIMEOUT_SECONDS = 5

_OVERRIDE_TOOLTIP = "Optional override; leave blank to use the 4yi gateway env."


def _model_widget(model_type: str):
    """Build the `model` widget: a dropdown of the caller's entitled models of
    `model_type`, fetched from the gateway. Falls back to a free-text field when
    the catalog can't be fetched at node-definition time (e.g. gateway env not
    set yet), so the node still loads and a name can be typed manually.

    Called from INPUT_TYPES (no access to the node's own api_base/api_key
    widgets there), so it reads the injected env for the gateway address.
    """
    try:
        base, key = resolve_gateway_config(os.environ)
    except GatewayError:
        return ("STRING", {"default": "", "tooltip": f"{model_type} model name."})
    models = _list_models(base, key, model_type)
    if not models:
        return ("STRING", {"default": "", "tooltip": f"{model_type} model name (gateway catalog unavailable)."})
    return (models, {"tooltip": f"Choose from your plan's {model_type} models."})


def _list_models(base: str, key: str, model_type: str):
    try:
        request = urllib.request.Request(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(request, timeout=MODELS_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
        return parse_model_list(body, model_type)
    except Exception:
        return []


def _auth_headers(url: str, base: str, key: str) -> dict:
    """Bearer only for gateway-origin URLs; never leak the token elsewhere."""
    if urlparse(url).netloc == urlparse(base).netloc:
        return {"Authorization": f"Bearer {key}"}
    return {}


async def _read_error(response: aiohttp.ClientResponse) -> str:
    try:
        body = await response.json(content_type=None)
        message = body.get("error", {}).get("message") if isinstance(body, dict) else None
        if message:
            return str(message)
    except Exception:
        pass
    return f"HTTP {response.status}"


async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict, key: str) -> dict:
    async with session.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}) as response:
        if response.status >= 400:
            raise GatewayError(f"gateway request failed: {await _read_error(response)}")
        return await response.json(content_type=None)


async def _get_json(session: aiohttp.ClientSession, url: str, key: str) -> dict:
    async with session.get(url, headers={"Authorization": f"Bearer {key}"}) as response:
        if response.status >= 400:
            raise GatewayError(f"gateway poll failed: {await _read_error(response)}")
        return await response.json(content_type=None)


async def _download_bytes(session: aiohttp.ClientSession, url: str, base: str, key: str) -> BytesIO:
    async with session.get(url, headers=_auth_headers(url, base, key)) as response:
        if response.status >= 400:
            raise GatewayError(f"artifact download failed: {await _read_error(response)}")
        return BytesIO(await response.read())


class FourYiGatewayImageGenerate:
    DESCRIPTION = "Generate images with a 4yi Gateway image model (OpenAI-compatible images API)."
    CATEGORY = "4yi Gateway"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    API_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": _model_widget("image"),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for the image model."}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536", "512x512"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Number of images to generate."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "control_after_generate": True,
                                 "tooltip": "Re-run control only; not sent to the gateway."}),
            },
            "optional": {
                "api_base": ("STRING", {"default": "", "tooltip": f"Gateway base URL (…/api/v1). {_OVERRIDE_TOOLTIP}"}),
                "api_key": ("STRING", {"default": "", "tooltip": f"Gateway API key. {_OVERRIDE_TOOLTIP}"}),
            },
        }

    async def generate(self, model, prompt, size, n, seed, api_base="", api_key=""):
        base, key = resolve_gateway_config(os.environ, override_base=api_base, override_key=api_key)
        if not str(model).strip():
            raise GatewayError("no image model selected")
        payload = build_image_payload(model=str(model).strip(), prompt=prompt, n=n, size=size)

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            body = await _post_json(session, f"{base}/images/generations", payload, key)
            tensors = []
            for kind, value in parse_image_entries(body):
                if kind == "b64":
                    data = BytesIO(base64.b64decode(value))
                else:
                    data = await _download_bytes(session, value, base, key)
                tensors.append(bytesio_to_image_tensor(data, mode="RGB"))
        return (torch.cat(tensors, dim=0),)


class FourYiGatewayVideoGenerate:
    DESCRIPTION = "Generate a video with a 4yi Gateway video model (async submit + poll)."
    CATEGORY = "4yi Gateway"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    API_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": _model_widget("video"),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for the video model."}),
                "duration_seconds": ("INT", {"default": 5, "min": 1, "max": 120}),
                "resolution": ("STRING", {"default": "", "tooltip": "Optional, model-specific (e.g. 720p, 1080p)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "control_after_generate": True,
                                 "tooltip": "Re-run control only; not sent to the gateway."}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "", "tooltip": "Public https image URL; switches to image-to-video."}),
                "extra_body": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Optional JSON object merged into the request's extra_body."}),
                "max_wait_seconds": ("INT", {"default": 1200, "min": 60, "max": 3600}),
                "api_base": ("STRING", {"default": "", "tooltip": f"Gateway base URL (…/api/v1). {_OVERRIDE_TOOLTIP}"}),
                "api_key": ("STRING", {"default": "", "tooltip": f"Gateway API key. {_OVERRIDE_TOOLTIP}"}),
            },
        }

    async def generate(self, model, prompt, duration_seconds, resolution, seed,
                       image_url="", extra_body="", max_wait_seconds=1200,
                       api_base="", api_key=""):
        base, key = resolve_gateway_config(os.environ, override_base=api_base, override_key=api_key)
        if not str(model).strip():
            raise GatewayError("no video model selected")
        payload = build_video_payload(
            model=str(model).strip(),
            prompt=prompt,
            duration_seconds=duration_seconds,
            resolution=resolution,
            image_url=image_url,
            extra_body_json=extra_body,
        )

        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            submitted = await _post_json(session, f"{base}/videos/generations", payload, key)
            job_id = str(submitted.get("id") or "").strip()
            if not job_id:
                raise GatewayError("gateway did not return a video job id")

            poll_url = f"{base}/videos/generations/{job_id}"

            async def fetch():
                return await _get_json(session, poll_url, key)

            max_attempts = max(1, int(max_wait_seconds) // POLL_INTERVAL_SECONDS)
            video_url = await poll_video_until_complete(
                fetch, sleep=asyncio.sleep,
                interval_seconds=POLL_INTERVAL_SECONDS, max_attempts=max_attempts,
            )
            data = await _download_bytes(session, video_url, base, key)
        return (InputImpl.VideoFromFile(data),)


NODE_CLASS_MAPPINGS = {
    "FourYiGatewayImageGenerate": FourYiGatewayImageGenerate,
    "FourYiGatewayVideoGenerate": FourYiGatewayVideoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FourYiGatewayImageGenerate": "4yi Image Generate (Gateway)",
    "FourYiGatewayVideoGenerate": "4yi Video Generate (Gateway)",
}
