# DSpine drafter on llama.cpp (SYCL, tensor parallel)

[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) is a DFlash-style block drafter for Qwen3.8-27B: 5 Qwen3 layers (hidden 5120), a block of 16 tokens, block-causal attention with an 8,192-token sliding window, and features from target layers 1/16/31/46/61. What makes it different is the **code bridge**: after every draft layer, each block position gets a gated message from its predecessor through a 1,024-dim code space (rank 512), and the last layer's write is redone exactly per candidate to score the next token ("exact-sender refinement").

As of this writing, llama.cpp doesn't support it. The patch in this repo adds it.

## What the patch adds

- **Converter** (`conversion/qwen.py`, `conversion/dspine.py`): maps `selected_prefix_bridge.*` to `dspine.*` tensors and the bridge settings to `<arch>.dspine.*` keys. It rejects unsupported variants (non-zero terminal projection, other write modes, unknown version markers). It also writes two derived F32 tensors used by the refinement: the last write matrix transposed, and its Gram matrix (1 MB).
- **Graph** (`src/models/dflash.cpp`):
  - the bridge write after every active layer;
  - the refinement table: for each block position and each top-16 candidate of its predecessor, the exact sender write, scored against the top-16 candidates of the position through the final norm and the output head.

  The table is packed into the same lattice layout as DFlash2's selector, so tree drafting, probabilistic draft sampling and verification work unchanged.
- **Rank-space refinement:** the refinement never builds the 5,120-wide vectors. Scores come from 512-dim products, with the RMS-norm denominator kept, so temperature-1 sampling stays exact. Against the direct path on identical weights it gives:
  - the same candidates in every cycle;
  - scores within 0.07 on a range of ±37;
  - 3 argmax flips in 8,400 rows, all near-ties.

  It isn't faster at this size (the draft pass is 6.5 ms either way), but it computes in F32 as the reference does.
- **Reduced head:** refinement reads head rows by target token id, so a drafter with its own 64K-token head (`d2t`) works. Build it with `scripts/make_draft_head.py`.

## Converting

```bat
python convert_hf_to_gguf.py <DSpine checkpoint dir> --target-model-dir <Qwen3.8-27B HF dir (config + tokenizer)> --outtype bf16 --no-lazy --outfile DSpine-b16-BF16.gguf
llama-quantize DSpine-b16-BF16.gguf DSpine-b16-Q4_K_M.gguf Q4_K_M
python scripts\make_draft_head.py freq --out freq.json <your text files>     (server running)
python scripts\make_draft_head.py head --freq freq.json --n 65536 --draft DSpine-b16-Q4_K_M.gguf --target Qwen3.8-27B-Q8_0.gguf
```

`--no-lazy` is needed because the converter checks the version markers and the zero terminal projection. **Use a quantized file.** On SYCL, the BF16 GGUF crashes the GPU driver on the second request, even with the bridge disabled; Q8_0 and Q4_K_M run fine.

## Results

### Controlled test (94K-token code prompt, temperature 1.0, 3 seeds, drafter only, look-up off, tree-12)

| Drafter | Tokens per cycle | t/s |
|---|---|---|
| DFlash2 Q4_K_M, 64K head | 4.64 | 66.9 |
| DSpine Q4_K_M, full head | 5.95 | 80.7 |
| **DSpine Q4_K_M, 64K head** | **5.99** | **83.0 (+24%)** |

At short context the two tie (90.6 vs 90.3 t/s). DSpine accepts about 5% more tokens per cycle there, but its 16-token draft pass costs more.

### Real agent sessions (look-up drafting on, temperature 1.0)

| Context | DFlash2 (656 requests) | DSpine (215 requests) | Change |
|---|---|---|---|
| 70–100K | 64.1 t/s | 65.2 t/s | +1.7% |
| 100–140K | 67.1 t/s | 68.2 t/s | +1.6% |
| 140K+ | 61.9 t/s | 73.5 t/s (31 requests) | +19% |
| **All** | **64.9 t/s** | **68.2 t/s** | **+5%** |

**Why real use gains much less than the benchmark:**
- DSpine accepts about 10% more tokens per cycle in real sessions, but each cycle costs 2–3 ms more, because it drafts 15 tokens instead of 7.
- Look-up drafting already handles the copy-heavy turns, so DSpine only helps on new text and reasoning.
- Long agent reasoning suits it less than the benchmark's code review.

**Caveats:**
- The DFlash2 sessions span a few earlier builds that ran slightly slower per cycle.
- The 140K+ bin is small, and session-to-session swings are larger than the drafter difference.

In short: a small but real win in normal use, and a large one on code-heavy prompts.
