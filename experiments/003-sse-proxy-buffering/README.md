# 003 — Streaming Lag Was the Proxy Buffer

> An external inference setup felt slow. GPU metrics were healthy, but tokens arrived in one big block after a pause. The proxy's read buffer was hiding real-time generation behind a fill-then-release pattern.

**Hardware:** Remote GPU server
**Model:** Qwen3.6-27B Q4_K_M
**Engine:** llama.cpp turboquant fork
**Date:** 2026-08-09

---

## Quick Start

If you're running a Python SSE proxy in front of llama-server, make sure your read buffer is small:

```python
# Before (bug): 64 KiB buffer batches tokens
CHUNK = 65536

# After (fix): 128-byte buffer streams tokens in real-time
CHUNK = 128
```

**Full proxy config:** [`configs/perkey_proxy.py`](configs/perkey_proxy.py) — the patched version with `CHUNK = 128`.

---

## The symptom

External inference felt slow. Reasoning tokens arrived in one big block after a long pause — not streamed token-by-token as expected. No errors in logs, health endpoint returned 200, GPU looked busy.

## The wrong suspicion

We checked GPU metrics first:

| Metric | Value | Status |
|---|---|---|
| GPU busy | 98% | working |
| VRAM | 30.4 / 32 GB | full but not OOM |
| Temperature | 61°C | no thermal throttle |
| CPU load | 2.85 | fine |

No thermal throttling, no memory pressure, no CPU bottleneck. The GPU was working fine.

## The real cause

The SSE proxy used a **64 KiB buffer** for reading upstream responses:

```python
# The bug:
for chunk in response.read(65536):  # waits to fill 64 KiB before yielding
    yield chunk
```

Python's chunked-response reader waits to fill that buffer before yielding tokens to the client. So tokens were produced in real-time by llama-server, but held back by the proxy — released in one burst when the buffer filled. The friend saw a long pause, then the entire reasoning block appeared at once.

## The fix

```python
# The fix:
CHUNK = 128  # small buffer = real-time streaming

for chunk in response.read(CHUNK):
    yield chunk
```

Tokens now stream as they're produced. The friend sees reasoning incrementally.

## Measured Results

**Date:** 2026-08-09

| Metric | Before fix | After fix |
|---|---|---|
| Buffer size | 64 KiB | 128 bytes |
| Streaming | batched (one big block) | real-time |
| GPU busy | 98% | 98% |
| Perceived speed | slow (long pause, then burst) | fast (incremental) |

## Troubleshooting

### Streaming feels laggy but GPU is healthy

**Symptom:** GPU busy >90%, no thermal throttle, no memory pressure. But tokens arrive in bursts instead of streaming.

**Cause:** Proxy or gateway buffering. Anything >1 KiB read buffer will batch tokens.

**Fix:** Check your proxy's read buffer size. Reduce to <1 KiB for real-time streaming.

### How to diagnose

1. Check GPU busy % — if >90%, the GPU is working
2. Check proxy buffer sizes — anything >1 KiB will batch tokens
3. Test with direct curl (bypasses proxy):
   ```bash
   curl -N http://127.0.0.1:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"...","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```
4. If direct curl streams fine but proxied doesn't → proxy buffering is the cause

---

## What transfers

1. **When streaming feels laggy but GPU metrics are healthy, suspect the proxy/transport layer.** The bottleneck is between the GPU and the user's screen, not in the GPU itself.

2. **Anything >1 KiB read buffer will batch tokens.** Python's `urllib3`, `requests`, and `http.client` all have default read buffers. Check your specific library.

3. **Test with direct curl first.** Bypasses the proxy and isolates the problem to either the model/engine or the transport layer.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/perkey_proxy.py`](configs/perkey_proxy.py) | Patched SSE proxy with `CHUNK = 128` |
| [`scripts/test-streaming.sh`](scripts/test-streaming.sh) | Test streaming behavior with direct curl vs proxied |
