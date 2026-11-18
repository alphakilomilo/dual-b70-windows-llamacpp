# Qwen3.8-27B on 2× Intel Arc Pro B70, native Windows (llama.cpp SYCL)

A complete, measured recipe for running **Qwen3.8-27B Q8_0** with **tensor parallelism across two Arc Pro B70 (32 GB)** on Windows 11. It covers:
- **speculative decoding** with the **DSpine** drafter (ported to llama.cpp here), a **12-node draft tree** and **16-token look-up drafting**;
- a patch set for llama.cpp;
- the launcher and config;
- a standalone repro for a oneAPI runtime memory leak that hits every multi-GPU tensor-parallel SYCL setup.

**In real use:** 67.5 t/s decode, averaged over 610 requests of agentic coding at 70K–190K context, temperature 1.0. That's up from 51.2 t/s with the October version of this recipe. Details below.

## Real-world results

Observed numbers from the server logs of **two full working sessions** with the current setup: **610 requests and 933K generated tokens** of agentic coding (long reasoning, tool calls, screenshots) at **temperature 1.0**.

| Context | Requests | **Decode t/s** | ms per cycle | Tokens per cycle |
|---|---|---|---|---|
| 70–100K | 192 | 63.9 | 65.0 | 4.16 |
| 100–140K | 296 | 68.1 | 68.2 | 4.64 |
| 140K+ | 121 | 70.3 | 71.5 | 5.03 |
| **All** | **610** | **67.5** | **68.1** | **4.60** |

How to read the table:
- **Tokens per cycle** is how many tokens each target pass emits: accepted drafts plus one. At temperature 1.0 it depends heavily on how predictable the text is, so single requests swing widely (from ~40 to 200+ t/s for copy-heavy edits).
- **Decode rises at high context** even though each cycle gets slower. Late in long agent sessions the model re-reads and edits code it has already seen, so drafts are accepted more often.
- **Prompt processing:** a fresh, uncached 94K prompt runs at ~1,225 t/s. Agent turns mostly reuse the prompt cache and process only the new tokens.
- **The October version** of this recipe (DFlash2, 7-token chain drafts) did 51.2 t/s over 1,030 requests, at 3.41 tokens per cycle and 66.6 ms per cycle.

---

## 1. Hardware and software

| | |
|---|---|
| GPUs | 2× Intel Arc Pro B70 32 GB (no peer-to-peer between them) |
| CPU / RAM | i7-14700K / 32 GB |
| OS | Windows 11 Pro (build 26200) |
| GPU driver | Intel Arc graphics driver 32.0.101.9175 |
| Compiler | Intel oneAPI DPC++/C++ 2026.2 |
| Level Zero | Level Zero SDK 1.33.1 |
| oneDNN | built from source with SYCL runtimes (`scripts/build-onednn.bat`, oneDNN main) |
| llama.cpp | upstream `9183a552e` (v1.0.0 + 22 commits) + `patches/llama.cpp-9183a552e-dual-b70.patch` |
| Model | [unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth) Q8_0 + `mmproj-BF16.gguf` |
| Drafter | [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16), converted and quantized to Q4_K_M with a 64K-token head ([docs/dspine.md](docs/dspine.md)) |

## 2. Setup

```bat
:: oneDNN (SYCL)
git clone https://github.com/uxlfoundation/oneDNN oneDNN-sycl
cd oneDNN-sycl && ..\scripts\build-onednn.bat && cd ..

:: llama.cpp + patch
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 9183a552e5fdcbdb5243f187f3a63204eca7dba9
git apply ..\patches\llama.cpp-9183a552e-dual-b70.patch
cd ..
scripts\build-llama.bat          :: edit the paths at the top first
keepalive\build.bat              :: optional, see section 5
```

Then:
1. Convert the DSpine drafter ([docs/dspine.md](docs/dspine.md)), put the models in `models\` (or edit `config\llm-server.conf`).
2. Exempt the server from Windows power throttling (as admin):
   ```bat
   powercfg /powerthrottling disable /path "<full path>\llama.cpp\build\bin\llama-server.exe"
   ```
3. Run `scripts\start-llm-server.bat`.

`GGML_SYCL_F16=ON` in the build is important: prefill is several times slower without it, and it's off by default.

## 3. Server flags (`config/llm-server.conf`)

| Flag | Why |
|---|---|
| `--split-mode tensor` | With the patched all-reduce: 85 ms vs 123 ms per step for layer split at 94K, before the kernel work below. |
| `--device SYCL0,SYCL1 --main-gpu 0 --fit off` | Both cards, no automatic fitting. |
| `--mmproj-device SYCL1` | GPU0 also drives the display. On GPU1 the vision encoder never spills to host memory: ~3 s instead of ~27 s per screenshot. |
| `--ctx-size 200000 --flash-attn on` | F16 KV was faster than q8_0 KV here. |
| `--batch-size 7680 --ubatch-size 2560` | 2560 was best: 1024 was slower, and 4096/5120 dropped prefill to ~480 t/s. |
| `--spec-type ngram-simple,draft-dflash` `--spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 16` | Look-up first: when the last 16 tokens appeared earlier in the context, the 16 that followed them are the draft. Otherwise the DSpine drafter builds a 12-node tree. Copy-heavy edits went from 143 to 213 t/s. |
| `-md DSpine-b16-Q4_K_M-v64k --spec-draft-n-max 16` | DSpine drafts a block of 16 (the server clamps 16 to 15 drafts). |
| `--spec-draft-sampling probabilistic` | Drafts are sampled, and verification keeps the output distribution unchanged at temperature 1.0. |
| `--load-mode none` | Otherwise the 29 GB GGUF stays resident in host RAM after offload. |
| `--cache-ram 12288` | Prompt cache for agent sessions. |
| `--temp 1.0 --min-p 0.0 --repeat-penalty 1.0` | Qwen3.8 thinking-mode sampling. Speculative decoding is lossless, so the drafter never changes outputs. |

## 4. Environment switches (`scripts/start-llm-server.bat`)

### Stock llama.cpp / oneAPI

| Variable | Effect |
|---|---|
| `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` | Level Zero only, no OpenCL enumeration. |
| `ZES_ENABLE_SYSMAN=1` | Real free-VRAM reporting. |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` | Lifts the 4 GB single-allocation cap (needed for big KV allocations at 200K). |
| `GGML_SYCL_DEV2DEV_MEMCPY=1`, `GGML_SYCL_USE_LEVEL_ZERO_API=1` | Device copies through Level Zero (+5% decode). |
| `GGML_SYCL_ENABLE_MKL_FA=0` | Keep the oneMKL attention path off so dispatch is unambiguous. |
| `GGML_SYCL_ENABLE_GRAPH=2`, `GGML_SYCL_GRAPH_MULTI_DEVICE=1` | SYCL graph replay for the draft context only. |

### Added by the patch

All patch switches are off unless set, so the patched build behaves like upstream by default.

| Variable | Set | What it does | Measured |
|---|---|---|---|
| `GGML_SYCL_PER_DEVICE_CONTEXT` | **1** | One SYCL context per GPU, both wrapping one shared Level Zero context. Works around the oneAPI leak in section 7 (still present in oneAPI 2026.2). | Leak 0.48 → 0.015 MB per token, same speed. |
| `GGML_SYCL_TP_DIRECT` | 1 | The tensor-parallel all-reduce has each GPU read the peer's pinned host buffer directly (ping-pong buffers), with no CPU memcpy. | Part of 123 → 85 ms. |
| `GGML_SYCL_TP_GPUFLAG` | 1 | The GPUs join through flags in host memory (spin with uncached reads) instead of one host wait per all-reduce (~130 per step). Aborts with "device join timed out" if it ever misses. | 99 → 85 ms. |
| `GGML_SYCL_TP_BF16_SYM` | 1 | Large all-reduces travel as bf16 (round to nearest even), and both GPUs add the same two rounded values, so the replicas stay bit-identical. | Prefill 720 → 976 t/s. |
| `GGML_SYCL_TP_Q8X=2`, `GGML_SYCL_TP_Q8X_BLOCK=16` | 2 / 16 | Prefill all-reduces travel as 8-bit values with a scale per 16. | Fresh 94K prefill +12%. |
| `GGML_SYCL_ADD_RMS_FUSE` | 1 | Fuses ADD + RMS_NORM + MUL. | Small prefill gain. |
| `LLAMA_TEST_DFLASH_TP_MIRROR_HEAD` | 1 | Keeps the full output head on both GPUs (the drafter reuses the target's head). | Required for DFlash/DSpine under TP. |
| `GGML_SYCL_FA_XMX` (+ `_STREAM`, `_NARROW_STREAM`, `_NARROW_96`, `_WIDE21_64`) | 1 | XMX flash attention for decode and verify batches of 1–24 query tokens (D=256, 6 query heads per KV head, F16 cache), including streamed variants for the wider tree verifies. | 85 → 73 ms at 94K; tree verifies at chain speed. |
| `GGML_SYCL_FA_ONEDNN` / `GGML_SYCL_FA_ONEDNN_SYNC=0` | 1 / 0 | oneDNN SDPA for prefill and the vision encoder (also unmasked), per-backend-context partition cache, no host wait after each call. | Prefill +27% at 28K, +37% at 94K; screenshots 6.3 → 2.9 s. |
| `LLAMA_DFLASH_PREFILL_TAIL` | 1 | Every draft layer uses a sliding window, so prompt rows outside it are skipped instead of fed to the drafter. Drafts are unchanged. | Fresh 94K prompt 983 → 1,069 t/s (DFlash2). |
| `LLAMA_SPEC_TREE=12`, `LLAMA_SPEC_TREE_MATCH=1` | 12 / 1 | Best-first 12-node draft tree over the drafter's candidate lattice. The target samples at each node and follows the matching child, so every emitted token is a target sample (lossless at any temperature). Tree attention masks and tree-aware recurrent state (gated delta net and conv) are part of the patch. | See section 6. |
| `LLAMA_TARGET_DUAL_SCHED=16` | 16 | A second cached target graph for verify batches of up to 16 tokens, so look-up and tree verifies don't rebuild each other's graphs. | 64.2 → 62.4 ms per cycle (with GPU top-k). |
| `LLAMA_GPU_TOPK=20` | 20 | Top-20 logits per verify row computed on the GPU; full 248K-entry rows are fetched only on demand. | Included above; 1,400 rows checked, 0 mismatches. |
| `LLAMA_DFLASH_DEV` | **0** | Keeps drafter features on host copies. Upstream's device-buffer path (on by default) was 2–2.7 ms per cycle slower under tensor parallel when first measured. | |

Always on (no switch) in the patched build:
- **SYCL `get_tensor_2d`.** Without it, ggml falls back to one synchronous copy per row. Under TP every KV row is split across both GPUs, so a 40 MiB drafter checkpoint took 250 ms (now 31 ms).
- **DSpine support:** converter, bridge, refinement lattice. See [docs/dspine.md](docs/dspine.md).
- **Reference-counted prompt checkpoints** and incremental drafter history in the server.
- **FP8 ops are refused by the SYCL backend.** It has no FP8 kernels, and upstream's new FP8 KV-cache tests otherwise abort.

Diagnostic or experimental switches stay off; they're documented in comments next to their code.

## 5. Windows specifics

- **Pin to the P-cores at high priority** (`start /high /affinity FFFF` on a 14700K). Tensor parallel has one host thread feeding both GPUs, and landing on an E-core costs ~2%.
- **Power throttling exemption** (`powercfg /powerthrottling disable /path ...`). It's per exe path, so redo it if you move the build.
- **GPU keepalive** (`keepalive/`). A headless B70 drops to runtime D3 ~30–40 s after the last request, and Windows evicts its ~28 GB; the next request then stalls ~19 s while it's paged back. A 256-byte fill per GPU every 10 s prevents it. The launcher starts it, and it exits with the server.

## 6. How the numbers moved

Time per speculative cycle at 94K context (DFlash2, 7 drafts + 1):

| Step | ms per cycle |
|---|---|
| Layer split | 123 |
| Tensor parallel, host-wait all-reduce | 99 |
| + direct exchange + GPU-flag join | 85 |
| + symmetric bf16 all-reduce | 83–85 (prefill 720 → 976 t/s) |
| + XMX attention | 73 |
| + XMX int8 mat-mul (upstream #29864) | 64 |
| + Q4_K_M drafter | 62.7 |

After that, the gains came from drafting more per cycle rather than from making cycles faster:

| Step (94K, temperature 1.0) | Effect |
|---|---|
| 64K-token drafter head (`scripts/make_draft_head.py`) | 61.7 → 59.6 ms per cycle, same acceptance |
| Probabilistic draft sampling | +8% decode |
| 16-token look-up drafting + 12-node tree | Copy-heavy edits 143 → 213 t/s |
| Second target graph + GPU top-k | 64.2 → 62.4 ms per cycle |
| DSpine instead of DFlash2 | +24% on the 94K benchmark, ~+5% in real sessions ([docs/dspine.md](docs/dspine.md)) |

### Drafter choice, measured on real sessions

Benchmarks overstate drafter gains on agent work. DSpine accepts ~10% more tokens per cycle than DFlash2 in real sessions, but drafts 15 tokens instead of 7, so each cycle costs 2–3 ms more. Net: ~+5% (64.9 → 68.2 t/s over 656 and 215 requests). Earlier, DFlash2 beat EAGLE-4 (2.95 vs 2.84 tokens per cycle) because it drafts a whole block in one pass.

### Tried, didn't help

- **Fused all-reduce kernel:** 2.2× slower.
- **Batched copies, shared activation quantization, multi-row Q8 GEMV:** all correct, none measurably faster. Launch overhead isn't the bottleneck; the GPUs run ahead of the host.
- **Pinning the drafter to one GPU:** crashes, because the drafter borrows the target's output head from the cross-GPU buffer.
- **Larger ubatch** (4096/5120): prefill drops to ~480 t/s.
- **Look-up drafts longer than 16:** verifies of 32+ tokens cost 108–140 ms.
- **DSpine refinement in 512-dim rank space:** exact, but no faster (refinement was never the bottleneck).
- **Lower temperature** is faster (more drafts accepted), but costs output variety.

## 7. The oneAPI multi-GPU leak (affects any SYCL tensor-parallel setup)

**Symptom:** in long sessions, decode crept from 64 to ~200 ms per cycle, **even after the context was reset**. The server's private memory reached 92 GB on a 32 GB machine, and Windows paged. Benchmarks under ~15 minutes don't show it.

**Cause:** launching the same kernel alternately on two GPUs of one SYCL context leaks host memory on every launch. `repro/oneapi-multigpu-kernel-cache-leak/alt.cpp` reproduces it without llama.cpp:

| 1M launches | Leak per launch |
|---|---|
| Same kernel, one GPU | ~5 B |
| **Alternating GPUs, shared context** | **166 B** |
| Different kernel name per GPU | ~9 B |
| Separate SYCL context per GPU | ~9 B |

`SYCL_CACHE_TRACE=7` shows a fast-cache `Kernel inserted` on every device switch. In the runtime's `kernel_program_cache.hpp`, `saveKernel()` calls `MFastKernelCache.try_emplace(name, FastKernelSubcacheWrapper(...))`. When the key already exists, the temporary wrapper is destroyed, and its destructor erases every entry for the context, including the other GPU's. So every device switch misses. Each miss also appends a `(kernel name, device)` record to `MProgramToFastKernelCacheKeyMap`, which is never pruned. Tensor parallel switches devices thousands of times per decode, which comes to ~0.5 MB per token. Both Level Zero adapters (v1 and v2) are affected, and `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD` doesn't help. Still present in oneAPI 2026.2.

**Workaround** (`GGML_SYCL_PER_DEVICE_CONTEXT=1`): every dpct queue gets its own per-device SYCL context, made with `make_context<level_zero>(native, {device}, ownership::keep)` from the platform default context's native Level Zero context.
- Each fast cache only ever sees one device.
- USM is still shared at the Level Zero level, so host buffers allocated through one GPU's context keep working on the other GPU. `repro/.../shared_ze.cpp` checks that kernels and copies work across them.
- The context-equality checks in the all-reduce become "same native context".

## 8. Verification method

All speed claims compare **time per cycle at temperature 1.0 over fixed seeds** on a 94K-token code prompt and an edit prompt, using the production launcher and config. At temperature 1.0, tokens/s for a single sample swings ±25% with the sampled text, while time per cycle is stable to ±0.5 ms.

Correctness checks:
- `test-backend-ops` on SYCL against the CPU for every touched op, including the tree ops;
- greedy probes against the layer-split reference;
- image turns and checkpoint-restore runs;
- copy tests: a 130-`===` JavaScript block copied verbatim and as `edit_file` tool calls, fresh and at 94K, across cache-reusing turns.

## License

MIT (see `LICENSE`). `patches/llama.cpp-9183a552e-dual-b70.patch` modifies [llama.cpp](https://github.com/ggml-org/llama.cpp), which is also MIT-licensed (Copyright (c) 2023-2026 The ggml authors); the code it changes remains under that license. DSpine weights are under Tencent's license (see the model card).

## Caveats

- One machine, one driver, one model. Other Intel GPUs (Xe, Xe2 iGPU, A-series) are untested.
- The patch targets `9183a552e`. Upstream has since refactored the DFlash drafters, the server slot scheduler and the batch API, so later commits need real porting work.
- The XMX attention kernels are specialised for D=256 / GQA 6 / F16 KV (Qwen3.8-27B); other shapes fall back to the stock kernels.
- Tree verification is wired for Qwen3.8's hybrid layout (gated delta net + attention) with one sequence (`--parallel 1`).
