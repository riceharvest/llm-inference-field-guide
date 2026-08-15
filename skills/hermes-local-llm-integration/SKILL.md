---
name: hermes-local-llm-integration
description: Use when connecting Hermes Agent to a local OpenAI-compatible LLM server or validating whether a local model is viable for real Hermes tool-use workflows, context growth, compression, and fallback routing.
version: 1.0.0
author: riceharvest contributors
license: MIT
metadata:
  hermes:
    tags: [hermes-agent, local-llm, tool-calling, context, routing]
    related_skills: [llama-cpp-serving-operator, inference-bottleneck-diagnostics]
---

# Hermes Local LLM Integration

## Overview

A local endpoint returning chat text is not enough. Hermes viability requires context fit, structured tools, multi-turn stability, reasoning/final-content handling, prompt caching, and useful behavior on real tasks.

Use the live Hermes documentation and CLI help as authority because configuration keys evolve: https://hermes-agent.nousresearch.com/docs

## When to Use

- Adding a llama.cpp, vLLM, SGLang, LM Studio, or Ollama OpenAI-compatible endpoint to Hermes.
- Diagnosing empty content, malformed tools, context errors, dead local ports, or slow agent loops.
- Comparing local models for Hermes workflows.
- Designing local-first fallback or escalation.

## Integration Sequence

1. **Audit reality before config.** Check listening ports and probe `/v1/models`; identify the actual process and model. Inspect Hermes' active profile/config path. Completion criterion: endpoint, provider profile, and live process agree.
2. **Verify direct API semantics.** Test plain chat, reasoning/final-content fields, and one request containing a tool schema. Completion criterion: structured tool call and usable final content are observed.
3. **Configure through the live CLI.** Use `hermes config`, `hermes config set`, or `hermes setup`; do not invent keys from old examples. Keep server context and Hermes context limits consistent. Completion criterion: `hermes config check` passes.
4. **Run a trivial Hermes smoke.** Use a fresh session and require exactly `ok`. Completion criterion: Hermes reaches the intended local endpoint without fallback.
5. **Run a real tool gate.** Ask Hermes to inspect a known file/repository using tools without modifications. Completion criterion: correct tool selection, valid arguments, grounded answer, and no fabricated paths.
6. **Run a controlled edit gate.** In a disposable worktree, make a tiny change and execute the narrowest verification. Completion criterion: artifact and real command output are parent-verified.
7. **Test multi-turn/context behavior.** Resume the same session, measure prompt/cache tokens, compression, decode, wall time, and context errors. Completion criterion: the model survives realistic accumulated context.
8. **Design fallback from failures.** Escalate on repeated malformed tools, context overflow, empty final content, or quality-gate failure—not merely because a task sounds hard. Completion criterion: fallback trigger and recovery are tested.

## Local Endpoint Audit

```bash
ss -tlnp
curl -sS --max-time 3 http://127.0.0.1:8080/v1/models
hermes config path
hermes config check
hermes tools list
```

Never assume a configured localhost URL is live. Do not expose local servers on non-loopback interfaces without authentication and an explicit requirement.

## Hermes Smoke Pattern

Use the exact provider/model names configured on the target machine:

```bash
hermes chat -q 'Reply with exactly ok'   --provider LOCAL_PROVIDER -m LOCAL_MODEL -Q
```

Then run a read-only tool task against a real repository. Direct API throughput is compatibility evidence; only a real `hermes chat` run supports a Hermes-agent claim.

## Metrics

Record server context, Hermes configured context, active prompt tokens, cached tokens, raw/effective input tok/s, completion tok/s, wall time, tool calls completed, final-content visibility, quality result, and failure reason.

## Context Discipline

Hermes system instructions and tool schemas can be large. Measure the actual first request; do not assume a model's nominal context is usable. Enable only toolsets the workflow needs, but never remove tools based on intuition—inspect usage logs first. Keep compression/fallback available before the hard limit.

## Reasoning and Token Caps

Some reasoning models spend substantial output budget before producing visible content. If `finish_reason` is length and content is empty, inspect reasoning tokens and token limits before blaming tool parsing. Validate both reasoning preservation and final answer visibility.

## Routing Discipline

- Local-first is useful only when the local model passes the task's quality gate.
- Route simple mechanical work locally; escalate unknown debugging, broad edits, or repeated local failures when a stronger provider is available.
- Prompt-only routing is soft policy. For hard routing, use explicit orchestration or separate profiles/processes.
- Validate delegated agents separately; stale local delegation URLs can break children while the parent works.

## Common Pitfalls

1. Configuring a dead port.
2. Provider/model name mismatch between Hermes and `/v1/models`.
3. Claiming tool compatibility from plain chat.
4. Server context and Hermes context disagreeing.
5. Tool schemas consuming more context than expected.
6. Local model emits inline tool tags but API does not convert them to structured calls.
7. Too-low output cap yields reasoning but empty content.
8. Changing tools/config without starting a fresh session when required.
9. Benchmarking direct curl instead of a real Hermes tool loop.
10. Treating a fast but ungrounded coding result as a pass.

## Verification Checklist

- [ ] Live endpoint and process identified
- [ ] Plain chat and structured tool-call API tests pass
- [ ] Hermes config check passes
- [ ] Fresh-session trivial smoke passes
- [ ] Real read-only tool workflow passes
- [ ] Controlled edit + test passes in disposable workspace
- [ ] Multi-turn context and compression tested
- [ ] Metrics and failure reasons retained
- [ ] Fallback/escalation tested
