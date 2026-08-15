# Contributing to llm-inference-field-guide

## Experiment format

Each experiment must be a standalone, copy-pasteable recipe. Use this template:

```
experiments/
  NNN-slug/
    README.md
    configs/
    scripts/
    results/
      YYYY-MM-DD-name.json
      YYYY-MM-DD-name.md
    patches/   # optional
```

## Required README sections

- **Quick Start** — exact copy-paste commands first
- **The symptom** — what broke and how it showed up
- **The fix** — exact config, flags, and why
- **Measured Results** — benchmark numbers with date
- **Troubleshooting** — common failures and fixes
- **What transfers** — general lessons that apply elsewhere
- **Files** — table of `configs/`, `scripts/`, `results/`, `patches/`

## Rules

- Use generic paths in documentation. No private paths, usernames, tokens, API keys, logs containing prompts, or machine state.
- One problem per experiment. No methodology fluff.
- Add a `results/*.json` with the raw benchmark matrix.
- Add a `scripts/smoke-*.sh` that proves the setup works in 30 seconds.
- If the experiment had a hotfix, add it to `patches/`.
