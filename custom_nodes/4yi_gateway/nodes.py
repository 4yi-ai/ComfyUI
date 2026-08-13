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
import logging
import os
import ssl
import time
import urllib.request
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import numpy as np
import torch
from PIL import Image

try:  # certifi ships with aiohttp/requests; prefer it so urllib TLS works even
    import certifi  # if the base image's system CA bundle is missing.
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover - fall back to the system trust store
    _SSL_CONTEXT = ssl.create_default_context()

logger = logging.getLogger("4yi_gateway")

from comfy_api.latest import InputImpl
from comfy_api_nodes.util.conversions import bytesio_to_image_tensor

try:
    from .gateway_client import (
        GatewayError,
        build_edit_fields,
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
        build_edit_fields,
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
MODELS_TIMEOUT_SECONDS = 4
MODELS_CACHE_TTL_SECONDS = 60

_OVERRIDE_TOOLTIP = "Optional override; leave blank to use the 4yi gateway env."

# Cache the /models result per (base, model_type). INPUT_TYPES runs on every
# /object_info call, so without a cache each catalog build would block on a
# fresh HTTP round-trip (once per node). On a fetch failure we keep serving the
# last good list so a transient blip doesn't empty a user's dropdown.
_models_cache: dict = {}


def _model_widget(model_type: str, image_input_only: bool = False):
    """Build the `model` widget: a dropdown of the caller's entitled models of
    `model_type`, fetched from the gateway. Falls back to a free-text field when
    the catalog can't be fetched at node-definition time (e.g. gateway env not
    set yet), so the node still loads and a name can be typed manually.

    image_input_only: for the image-to-video node, list only video models the
    gateway marks as accepting an input image, so a text-to-video model is never
    offered for an i2v workflow.

    Called from INPUT_TYPES (no access to the node's own api_base/api_key
    widgets there), so it reads the injected env for the gateway address.
    """
    try:
        base, key = resolve_gateway_config(os.environ)
    except GatewayError:
        logger.warning("4yi model catalog: gateway env not configured; %s field is free-text", model_type)
        return ("STRING", {"default": "", "tooltip": f"{model_type} model name."})
    models = _list_models(base, key, model_type, image_input_only)
    if not models:
        return ("STRING", {"default": "", "tooltip": f"{model_type} model name (gateway catalog unavailable)."})
    return (models, {"tooltip": f"Choose from your plan's {model_type} models."})


def _list_models(base: str, key: str, model_type: str, image_input_only: bool = False):
    cache_key = (base, model_type, image_input_only)
    cached = _models_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < MODELS_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        request = urllib.request.Request(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(request, timeout=MODELS_TIMEOUT_SECONDS, context=_SSL_CONTEXT) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = parse_model_list(body, model_type, image_input_only)
        _models_cache[cache_key] = (now, models)
        return models
    except Exception as error:  # log once per failure; keep serving stale list if we have one
        logger.warning("4yi model catalog: GET %s/models failed for %s: %s", base, model_type, error)
        return cached[1] if cached else []


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


def _image_to_png_bytes(image) -> bytes:
    """First image of a ComfyUI IMAGE batch ([B,H,W,C] float 0-1) -> PNG bytes."""
    array = (image[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


async def _post_multipart(session: aiohttp.ClientSession, url: str, fields: dict, image_png: bytes, key: str) -> dict:
    form = aiohttp.FormData()
    for name, value in fields.items():
        form.add_field(name, value)
    form.add_field("image", image_png, filename="image.png", content_type="image/png")
    async with session.post(url, data=form, headers={"Authorization": f"Bearer {key}"}) as response:
        if response.status >= 400:
            raise GatewayError(f"gateway request failed: {await _read_error(response)}")
        return await response.json(content_type=None)


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


class FourYiGatewayImageEdit:
    DESCRIPTION = "Edit/restyle an uploaded image with a 4yi Gateway image model (OpenAI-compatible images/edits)."
    CATEGORY = "4yi Gateway"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    API_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": _model_widget("image"),
                "image": ("IMAGE", {"tooltip": "Source image to edit (from Load Image)."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "How to edit the image, e.g. 换成纯白背景 / 转成日系插画风。"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536", "512x512"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "control_after_generate": True,
                                 "tooltip": "Re-run control only; not sent to the gateway."}),
            },
            "optional": {
                "api_base": ("STRING", {"default": "", "tooltip": f"Gateway base URL (…/api/v1). {_OVERRIDE_TOOLTIP}"}),
                "api_key": ("STRING", {"default": "", "tooltip": f"Gateway API key. {_OVERRIDE_TOOLTIP}"}),
            },
        }

    async def generate(self, model, image, prompt, size, n, seed, api_base="", api_key=""):
        base, key = resolve_gateway_config(os.environ, override_base=api_base, override_key=api_key)
        if not str(model).strip():
            raise GatewayError("no image model selected")
        fields = build_edit_fields(model=str(model).strip(), prompt=prompt, n=n, size=size)
        image_png = _image_to_png_bytes(image)

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            body = await _post_multipart(session, f"{base}/images/edits", fields, image_png, key)
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
                "image": ("IMAGE", {"tooltip": "上传首帧图 = 图生视频(从 Load Image 连入);留空 = 纯文生视频。优先于 image_url。"}),
                "image_url": ("STRING", {"default": "", "tooltip": "Public https image URL; switches to image-to-video. Ignored when an image is connected."}),
                "extra_body": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Optional JSON object merged into the request's extra_body."}),
                "max_wait_seconds": ("INT", {"default": 1200, "min": 60, "max": 3600}),
                "api_base": ("STRING", {"default": "", "tooltip": f"Gateway base URL (…/api/v1). {_OVERRIDE_TOOLTIP}"}),
                "api_key": ("STRING", {"default": "", "tooltip": f"Gateway API key. {_OVERRIDE_TOOLTIP}"}),
            },
        }

    async def generate(self, model, prompt, duration_seconds, resolution, seed,
                       image=None, image_url="", extra_body="", max_wait_seconds=1200,
                       api_base="", api_key=""):
        base, key = resolve_gateway_config(os.environ, override_base=api_base, override_key=api_key)
        if not str(model).strip():
            raise GatewayError("no video model selected")
        # An uploaded first frame (from Load Image) takes precedence over a URL:
        # inline it as a base64 data URL, which the gateway i2v path accepts
        # (DashScope/Bailian consume data:image/…;base64 directly; the SSRF
        # allowlist only vets http(s) URLs, so a data URL passes through).
        first_frame = str(image_url).strip()
        if image is not None:
            first_frame = "data:image/png;base64," + base64.b64encode(_image_to_png_bytes(image)).decode("ascii")
        payload = build_video_payload(
            model=str(model).strip(),
            prompt=prompt,
            duration_seconds=duration_seconds,
            resolution=resolution,
            image_url=first_frame,
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


class FourYiGatewayImageToVideo(FourYiGatewayVideoGenerate):
    DESCRIPTION = "Image-to-video with a 4yi Gateway i2v/r2v model: upload a first frame + prompt."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Only video models the gateway marks as accepting an input image
                # (i2v/first-frame/reference); text-to-video models are excluded.
                "model": _model_widget("video", image_input_only=True),
                "image": ("IMAGE", {"tooltip": "首帧图(从 Load Image 连入):以这张图为基础生成视频。"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for the video model."}),
                "duration_seconds": ("INT", {"default": 5, "min": 1, "max": 120}),
                "resolution": ("STRING", {"default": "", "tooltip": "Optional, model-specific (e.g. 720p, 1080p)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "control_after_generate": True,
                                 "tooltip": "Re-run control only; not sent to the gateway."}),
            },
            "optional": {
                "extra_body": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Optional JSON object merged into the request's extra_body."}),
                "max_wait_seconds": ("INT", {"default": 1200, "min": 60, "max": 3600}),
                "api_base": ("STRING", {"default": "", "tooltip": f"Gateway base URL (…/api/v1). {_OVERRIDE_TOOLTIP}"}),
                "api_key": ("STRING", {"default": "", "tooltip": f"Gateway API key. {_OVERRIDE_TOOLTIP}"}),
            },
        }

    # generate() is inherited from FourYiGatewayVideoGenerate: ComfyUI passes
    # inputs by keyword, so the required `image` maps to the same param and is
    # inlined as the i2v first frame.


NODE_CLASS_MAPPINGS = {
    "FourYiGatewayImageGenerate": FourYiGatewayImageGenerate,
    "FourYiGatewayImageEdit": FourYiGatewayImageEdit,
    "FourYiGatewayVideoGenerate": FourYiGatewayVideoGenerate,
    "FourYiGatewayImageToVideo": FourYiGatewayImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FourYiGatewayImageGenerate": "4yi Image Generate (Gateway)",
    "FourYiGatewayImageEdit": "4yi Image Edit (Gateway)",
    "FourYiGatewayVideoGenerate": "4yi Video Generate (Gateway)",
    "FourYiGatewayImageToVideo": "4yi Image-to-Video (Gateway)",
}
