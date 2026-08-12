# 4yi Gateway nodes

Two nodes that route generation through the 4yi Gateway instead of local
model weights, letting ComfyUI run on a CPU-only container:

- **4yi Image Generate (Gateway)** — `POST {base}/images/generations`
  (OpenAI-compatible, synchronous). Outputs `IMAGE`.
- **4yi Video Generate (Gateway)** — `POST {base}/videos/generations`
  (async: 202 + job id), polls `GET {base}/videos/generations/{id}` every 5s,
  then downloads the artifact. Outputs `VIDEO` (feed into Save Video).
  Setting `image_url` (public https URL) switches to image-to-video and is
  sent as `extra_body.first_frame`; the `extra_body` field accepts a JSON
  object for provider-specific parameters.

## Configuration

Zero-config on the 4yi App Platform — these env vars are injected per install:

| Env | Meaning |
| --- | --- |
| `IMAGE_API_BASE` / `OPENAI_API_BASE` | Gateway base URL (`https://<origin>/api/v1`) |
| `IMAGE_API_KEY` / `OPENAI_API_KEY` | Per-install gateway token |
| `IMAGE_MODEL` | Default image model slot |
| `VIDEO_MODEL` | Default video model slot |

Every node also has optional `model` / `api_base` / `api_key` fields that
override the env for that node only. The API key is only ever sent to the
configured gateway origin (including artifact downloads); other hosts are
fetched without credentials.

The `seed` widget is a re-run control (`control_after_generate`) so you can
re-roll a generation without changing the prompt — it is not sent to the
gateway.

## Tests

Pure request/response logic lives in `gateway_client.py` (no ComfyUI/torch
imports) and is covered by:

```bash
python3 -m pytest custom_nodes/4yi_gateway/tests -q
```
