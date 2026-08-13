"""Tests for the pure-logic layer of the 4yi Gateway custom nodes.

No ComfyUI / torch imports here: gateway_client must stay importable and
testable on a bare Python + aiohttp environment.
"""

import asyncio

import pytest

from gateway_client import (
    GatewayError,
    build_edit_fields,
    build_image_payload,
    build_video_payload,
    interpret_video_poll,
    parse_image_entries,
    parse_model_list,
    poll_video_until_complete,
    resolve_gateway_config,
)


# ── image edit (multipart form fields) ───────────────────────────────────────

def test_build_edit_fields_basic():
    fields = build_edit_fields(model="gemini-3.1-flash-image", prompt="把背景换成纯白", n=1, size="1024x1024")
    assert fields == {
        "model": "gemini-3.1-flash-image",
        "prompt": "把背景换成纯白",
        "n": "1",
        "size": "1024x1024",
        "response_format": "b64_json",
    }


def test_build_edit_fields_omits_auto_size_clamps_n_and_stringifies():
    fields = build_edit_fields(model="m", prompt="p", n=99, size="auto")
    assert "size" not in fields
    assert fields["n"] == "10"        # clamped, and a string (multipart form value)
    assert build_edit_fields(model="m", prompt="p", n=0, size="auto")["n"] == "1"


def test_build_edit_fields_requires_prompt():
    with pytest.raises(GatewayError, match="prompt"):
        build_edit_fields(model="m", prompt="   ", n=1, size="auto")


# ── model catalog (/models) ──────────────────────────────────────────────────

def test_parse_model_list_filters_by_type_top_level_type_field():
    body = {"data": [
        {"id": "gemini-3.1-flash-image", "type": "image"},
        {"id": "happyhorse-1.1-t2v", "type": "video"},
        {"id": "gpt-5.5", "type": "chat"},
        {"id": "gpt-5-image", "type": "image"},
    ]}
    assert parse_model_list(body, "image") == ["gemini-3.1-flash-image", "gpt-5-image"]
    assert parse_model_list(body, "video") == ["happyhorse-1.1-t2v"]


def test_parse_model_list_image_input_only_keeps_image_capable_video_models():
    body = {"data": [
        {"id": "happyhorse-1.1-t2v", "type": "video", "video_image_input": False},
        {"id": "klingai/kling-v3-i2v", "type": "video", "video_image_input": True},
        {"id": "klingai/kling-v3-omni-ref2v", "type": "video", "video_image_input": True},
        {"id": "gpt-4o", "type": "chat"},
    ]}
    # Without the flag: all video models.
    assert parse_model_list(body, "video") == [
        "happyhorse-1.1-t2v", "klingai/kling-v3-i2v", "klingai/kling-v3-omni-ref2v",
    ]
    # image_input_only: drops the t2v model, keeps i2v/r2v.
    assert parse_model_list(body, "video", image_input_only=True) == [
        "klingai/kling-v3-i2v", "klingai/kling-v3-omni-ref2v",
    ]


def test_parse_model_list_image_input_only_absent_flag_is_excluded():
    # A video model with no video_image_input field is treated as not image-capable.
    body = {"data": [{"id": "legacy-video", "type": "video"}]}
    assert parse_model_list(body, "video", image_input_only=True) == []


def test_parse_model_list_accepts_model_type_and_name_aliases():
    body = {"data": [
        {"name": "wan-2.7", "model_type": "video"},
        {"name": "img-x", "model_type": "image"},
    ]}
    assert parse_model_list(body, "video") == ["wan-2.7"]


def test_parse_model_list_empty_or_missing_data():
    assert parse_model_list({}, "image") == []
    assert parse_model_list({"data": []}, "video") == []
    assert parse_model_list({"data": [{"id": "x", "type": "chat"}]}, "image") == []


# ── config resolution ────────────────────────────────────────────────────────

def test_resolve_config_reads_image_env_and_strips_trailing_slash():
    env = {"IMAGE_API_BASE": "https://gw.example.com/api/v1/", "IMAGE_API_KEY": "xclaw-bsl-abc"}
    base, key = resolve_gateway_config(env)
    assert base == "https://gw.example.com/api/v1"
    assert key == "xclaw-bsl-abc"


def test_resolve_config_falls_back_to_openai_env():
    env = {"OPENAI_API_BASE": "https://gw.example.com/api/v1", "OPENAI_API_KEY": "xck-123"}
    base, key = resolve_gateway_config(env)
    assert base == "https://gw.example.com/api/v1"
    assert key == "xck-123"


def test_resolve_config_widget_override_wins():
    env = {"IMAGE_API_BASE": "https://env.example.com/api/v1", "IMAGE_API_KEY": "env-key"}
    base, key = resolve_gateway_config(env, override_base="https://widget.example.com/api/v1/", override_key="widget-key")
    assert base == "https://widget.example.com/api/v1"
    assert key == "widget-key"


def test_resolve_config_missing_base_raises():
    with pytest.raises(GatewayError, match="IMAGE_API_BASE"):
        resolve_gateway_config({"IMAGE_API_KEY": "k"})


def test_resolve_config_missing_key_raises():
    with pytest.raises(GatewayError, match="IMAGE_API_KEY"):
        resolve_gateway_config({"IMAGE_API_BASE": "https://gw.example.com/api/v1"})


# ── image payload / response ─────────────────────────────────────────────────

def test_build_image_payload_basic():
    payload = build_image_payload(model="gpt-image-1", prompt="a red fox", n=2, size="1024x1024")
    assert payload == {
        "model": "gpt-image-1",
        "prompt": "a red fox",
        "n": 2,
        "size": "1024x1024",
        "response_format": "b64_json",
    }


def test_build_image_payload_omits_auto_size_and_clamps_n():
    payload = build_image_payload(model="m", prompt="p", n=99, size="auto")
    assert "size" not in payload
    assert payload["n"] == 10
    assert build_image_payload(model="m", prompt="p", n=0, size="auto")["n"] == 1


def test_build_image_payload_requires_prompt():
    with pytest.raises(GatewayError, match="prompt"):
        build_image_payload(model="m", prompt="   ", n=1, size="auto")


def test_parse_image_entries_b64_and_url():
    body = {"data": [{"b64_json": "aGk="}, {"url": "https://cdn.example.com/x.png"}]}
    entries = parse_image_entries(body)
    assert entries == [("b64", "aGk="), ("url", "https://cdn.example.com/x.png")]


def test_parse_image_entries_empty_raises():
    with pytest.raises(GatewayError, match="no image data"):
        parse_image_entries({"data": []})
    with pytest.raises(GatewayError, match="no image data"):
        parse_image_entries({})


# ── video payload ────────────────────────────────────────────────────────────

def test_build_video_payload_t2v():
    payload = build_video_payload(model="veo-3.1", prompt="a drone shot", duration_seconds=5, resolution="720p")
    assert payload == {
        "model": "veo-3.1",
        "prompt": "a drone shot",
        "duration_seconds": 5,
        "mode": "text_to_video",
        "resolution": "720p",
    }


def test_build_video_payload_omits_empty_resolution():
    payload = build_video_payload(model="m", prompt="p", duration_seconds=5, resolution="")
    assert "resolution" not in payload


def test_build_video_payload_i2v_via_image_url():
    payload = build_video_payload(model="kling-v3-i2v", prompt="p", duration_seconds=5, resolution="", image_url="https://cdn.example.com/frame.png")
    assert payload["mode"] == "image_to_video"
    # First frame is sent in BOTH channels so it works across providers:
    # Bailian reads __studio_reference_images, OpenRouter reads first_frame.
    assert payload["extra_body"] == {
        "first_frame": "https://cdn.example.com/frame.png",
        "__studio_reference_images": ["https://cdn.example.com/frame.png"],
    }


def test_build_video_payload_merges_extra_body_json():
    payload = build_video_payload(
        model="m", prompt="p", duration_seconds=5, resolution="",
        image_url="https://cdn.example.com/frame.png",
        extra_body_json='{"aspect_ratio": "16:9"}',
    )
    assert payload["extra_body"] == {
        "first_frame": "https://cdn.example.com/frame.png",
        "__studio_reference_images": ["https://cdn.example.com/frame.png"],
        "aspect_ratio": "16:9",
    }


def test_build_video_payload_rejects_bad_extra_body():
    with pytest.raises(GatewayError, match="extra_body"):
        build_video_payload(model="m", prompt="p", duration_seconds=5, resolution="", extra_body_json="{nope")
    with pytest.raises(GatewayError, match="extra_body"):
        build_video_payload(model="m", prompt="p", duration_seconds=5, resolution="", extra_body_json='["not-an-object"]')


def test_build_video_payload_validates_duration():
    with pytest.raises(GatewayError, match="duration"):
        build_video_payload(model="m", prompt="p", duration_seconds=0, resolution="")
    with pytest.raises(GatewayError, match="duration"):
        build_video_payload(model="m", prompt="p", duration_seconds=121, resolution="")


# ── video polling ────────────────────────────────────────────────────────────

def test_interpret_video_poll_states():
    assert interpret_video_poll({"status": "running"}) is None
    assert interpret_video_poll({"status": "completed", "video_url": "https://gw/api/v1/videos/generations/vg_1/artifact"}) \
        == "https://gw/api/v1/videos/generations/vg_1/artifact"
    with pytest.raises(GatewayError, match="boom"):
        interpret_video_poll({"status": "failed", "failure_message": "boom"})
    with pytest.raises(GatewayError, match="video_url"):
        interpret_video_poll({"status": "completed"})


def test_poll_video_until_complete_returns_url():
    responses = [{"status": "running"}, {"status": "running"}, {"status": "completed", "video_url": "https://gw/v.mp4"}]
    sleeps = []

    async def fetch():
        return responses.pop(0)

    async def sleep(seconds):
        sleeps.append(seconds)

    url = asyncio.run(poll_video_until_complete(fetch, sleep=sleep, interval_seconds=5, max_attempts=10))
    assert url == "https://gw/v.mp4"
    assert sleeps == [5, 5]


def test_poll_video_until_complete_times_out():
    async def fetch():
        return {"status": "running"}

    async def sleep(_seconds):
        pass

    with pytest.raises(GatewayError, match="timed out"):
        asyncio.run(poll_video_until_complete(fetch, sleep=sleep, interval_seconds=1, max_attempts=3))


def test_poll_video_until_complete_propagates_failure():
    async def fetch():
        return {"status": "failed", "failure_message": "provider exploded"}

    async def sleep(_seconds):
        pass

    with pytest.raises(GatewayError, match="provider exploded"):
        asyncio.run(poll_video_until_complete(fetch, sleep=sleep, interval_seconds=1, max_attempts=3))
