# 004 — Serverless Worker Drift

> A serverless inference setup entered a broken state after repeated template updates. Worker config and platform config drifted apart, causing endless recycles. The only reliable fix was to delete the worker and recreate it from a finalized config.

**Hardware:** Serverless GPU endpoint
**Model:** Qwen3.6-27B MTP NVFP4
**Engine:** vLLM 0.25.1
**Date:** 2026-08-08

---

## Quick Start

To avoid this entirely:

1. **Finalize your template locally first.** Don't iterate through the platform UI.
2. **Push once.** Don't patch the worker group incrementally.
3. **After creating a worker, verify the GPU matches your filters:**
   ```bash
   vastai show instances --raw | jq '.[] | {id, gpu_name, num_gpus, actual_status}'
   ```
4. **If your worker is the wrong GPU or stuck in `update_rebooting`, delete the worker group and recreate it.** Don't try to patch it back to health.

---

## The symptom

After several template tweaks, the endpoint entered "update_rebooting" and never came out. Worker kept recycling. Configs kept resetting to defaults.

## What happened

1. Updated the template to fix the model config
2. The worker rebooted to pick up the new template
3. Another update reset the GPU search filters to generic defaults
4. The new worker spawned with the wrong GPU
5. Fixed the filters → another reboot → another reset
6. Infinite loop: worker config and platform config were permanently out of sync

## The fix

Delete the worker group entirely and recreate it once with:
- Final template hash
- Final launch arguments
- Full search filters (GPU count, VRAM, bid pricing)

Accept the worker destruction — a clean slate beats incremental updates.

Update the endpoint with conservative autoscaling parameters once the new worker is healthy.

```bash
vastai update endpoint <your-endpoint-id> \
  --min_load 0 --max_workers 1 \
  --cold_workers 1 --cold_mult 1 \
  --target_util 0.9 \
  --inactivity_timeout 300
```

## Measured Results

**Date:** 2026-08-08

| Metric | During drift | After fix |
|---|---|---|
| Worker state | `update_rebooting` (hours) | `Ready` |
| GPU match | wrong GPU | correct GPU |
| Endpoint health | stuck | 200 OK |

## Troubleshooting

### Worker stuck in `update_rebooting` for more than 10 minutes

**Symptom:** Worker keeps recycling.

**Cause:** Config drift. Each update resets different parts of the config.

**Fix:** Delete and recreate with the final config. Don't patch incrementally.

### Wrong GPU after template update

**Symptom:** You set specific GPU filters, but the worker spawns with a different GPU.

**Cause:** The update reset the search filters to generic defaults.

**Fix:** Verify the worker's actual GPU matches your intent. If not, delete and recreate with correct filters.

---

## What transfers

1. **On serverless GPU platforms, config updates are destructive, not rolling.** Don't patch — recreate.

2. **If config drifts, you get infinite recycle loops.** The worker keeps rebooting, but each update resets something else.

3. **Always verify the worker's actual GPU matches your filters after each update.** Don't assume filters survived intact.

4. **A clean worker group beats a half-patched one.** If stuck, destroy it instead of trying to recover it.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/workergroup-search-params.json`](configs/workergroup-search-params.json) | Example search filters |
| [`scripts/verify-worker.sh`](scripts/verify-worker.sh) | Verify worker GPU matches your filters |
