# Qwen3.8-27B on 2× Intel Arc Pro B70, native Windows (llama.cpp SYCL)

A complete, measured recipe for running **Qwen3.8-27B Q8_0** with **tensor parallelism across two Arc Pro B70 (32 GB)** and the **DFlash2** speculative drafter on Windows 11. It includes a patch set for llama.cpp, the launcher and config, and a standalone repro for a oneAPI runtime memory leak that hits every multi-GPU tensor-parallel SYCL setup.

**In real use:** 51 t/s decode and ~840 t/s prompt processing on agent turns, averaged over 1,030 requests at 15K–193K context, temperature 1.0. Details below.

## Real-world results

These are observed numbers from the server logs of **two full working sessions** with this exact setup. Together they cover **1,030 requests and 1.68M generated tokens** of agentic coding (long reasoning, tool calls, screenshots) at **temperature 1.0**, at 15K–193K context, ~14 hours of uptime in total.

| Context | Requests | **Decode t/s** | ms per cycle | Tokens per cycle | Draft acceptance | **Prompt processing t/s** (token-weighted) | Median turn PP t/s | Median new tokens per turn |
|---|---|---|---|---|---|---|---|---|
| < 40K | 15 | 49.1 | 57.7 | 2.83 | 26% | 1,054 | 1,047 | 6,041 |
| 40–80K | 201 | 46.6 | 62.5 | 2.91 | 27% | 963 | 895 | 4,529 |
| 80–120K | 387 | 48.1 | 65.2 | 3.13 | 30% | 820 | 808 | 4,256 |
| 120–160K | 399 | 51.9 | 68.4 | 3.55 | 36% | 794 | 726 | 3,739 |
| 160K+ | 28 | 61.0 | 70.9 | 4.32 | 47% | 702 | 760 | 7,532 |
| **All** | **1,030** | **51.2** | **66.6** | **3.41** | **34%** | **838** | **781** | 4,236 |

How to read the table:
- **Decode per request** (requests of 200+ tokens) ranged from 39 t/s (10th percentile) through a 47 t/s median to 62 t/s (90th percentile). At temperature 1.0 it depends heavily on how predictable the text is.
- **Decode rises at high context** even though each cycle gets slower (57.7 → 70.9 ms). Late in long agent sessions the model re-reads and edits code it has already seen, so drafts are accepted more often. Time per cycle is the hardware-side number; tokens per cycle reflects the workload.
- **Prompt processing** is per agent turn (only the new tokens; the rest comes from the prompt cache), counting turns of at least 1,000 new tokens. A fresh, uncached 94K prompt runs at ~1,070 t/s.
- **Host memory** was 77.8 GB of private memory after 5.3 hours, from a typical ~65 GB at start (~2.5 GB/hour). Time per cycle depends only on context, not on uptime. Before the leak fix in section 7, it grew ~18 GB/hour and decode slowed to 200 ms per cycle.
- **Starting point:** the same machine did ~30 t/s in real sessions and slowed down as a session went on.

---

## 1. Hardware and software

| | |
|---|---|
| GPUs | 2× Intel Arc Pro B70 32 GB (no peer-to-peer between them) |
| CPU / RAM | i7-14700K / 32 GB |
| OS | Windows 11 Pro (build 26200) |
| GPU driver | Intel Arc Pro graphics driver 32.0.101.9150 |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.0 (build 20260617) |
| Level Zero | Level Zero SDK 1.33.1 |
| oneDNN | built from source with SYCL runtimes (`scripts/build-onednn.bat`, oneDNN main at `6e2d02e2b`) |
| llama.cpp | upstream `d81d0d8fb` ("Multi-column on Intel XMX matrix engines refactored (#29864)") + `patches/llama.cpp-d81d0d8fb-dual-b70.patch` |
| Model | [unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth) Q8_0 + `mmproj-BF16.gguf` |
| Drafter | [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) Q4_K_M |

## 2. Setup

```bat
:: oneDNN (SYCL)
git clone https://github.com/uxlfoundation/oneDNN oneDNN-sycl
cd oneDNN-sycl && ..\scripts\build-onednn.bat && cd ..

:: llama.cpp + patch
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout d81d0d8fb95e52a66992bcfc212b1891b4d2f94a
git apply ..\patches\llama.cpp-d81d0d8fb-dual-b70.patch
cd ..
scripts\build-llama.bat          :: edit the paths at the top first
keepalive\build.bat              :: optional, see section 5
```

Then:
1. Put the models in `models\` (or edit `config\llm-server.conf`).
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
| `--ctx-size 200000 --flash-attn on` | F16 KV was faster than q8_0 KV here. |
| `--batch-size 7680 --ubatch-size 2560` | 2560 was best: 1024 was slower, and 4096/5120 dropped prefill to ~480 t/s. |
| `--spec-type draft-dflash -md DFlash2-Q4_K_M --spec-draft-n-max 8` | DFlash2 drafts a block of 8, so 7 drafts per cycle (the server clamps 8 to 7). Q4_K_M is 1.3 ms per cycle faster than Q8_0 with the same acceptance. |
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

### Added by the patch

All patch switches are off unless set, so the patched build behaves like upstream by default.

| Variable | Set | What it does | Measured |
|---|---|---|---|
| `GGML_SYCL_PER_DEVICE_CONTEXT` | **1** | One SYCL context per GPU, both wrapping one shared Level Zero context. Works around the oneAPI leak in section 7. | Leak 0.48 → 0.015 MB per token, same speed. |
| `GGML_SYCL_TP_DIRECT` | 1 | The tensor-parallel all-reduce has each GPU read the peer's pinned host buffer directly (ping-pong buffers), with no CPU memcpy. | Part of 123 → 85 ms. |
| `GGML_SYCL_TP_GPUFLAG` | 1 | The GPUs join through flags in host memory (spin with uncached reads) instead of one host wait per all-reduce (~130 per step). Aborts with "device join timed out" if it ever misses. | 99 → 85 ms. |
| `GGML_SYCL_TP_BF16_SYM` | 1 | Large all-reduces travel as bf16 (round to nearest even), and both GPUs add the same two rounded values, so the replicas stay bit-identical. The stock asymmetric bf16 path let them drift; lossless F32 cost 26% prefill. | Prefill 720 → 976 t/s. |
| `LLAMA_TEST_DFLASH_TP_MIRROR_HEAD` | 1 | Keeps the full output head on both GPUs (DFlash2 reuses the target's head). | Required for DFlash2 under TP. |
| `GGML_SYCL_FA_XMX` | 1 | XMX split-K flash attention for 1–8 query tokens (D=256, 6 query heads per KV head, F16 cache). Used for decode and the speculative verify batch. The TILE kernel it replaces grouped the 6 heads in pairs and ran on the vector units. | 85 → 73 ms at 94K. |
| `GGML_SYCL_FA_ONEDNN` / `GGML_SYCL_FA_ONEDNN_SYNC=0` | 1 / 0 | oneDNN SDPA for prefill, per-backend-context partition cache, no host wait after each call. | Prefill +27% at 28K, +37% at 94K. |
| `GGML_SYCL_Q8_GLU_FUSION`, `GGML_SYCL_Q8_GLU_SPLIT_EPILOGUE` | 1 | Share the activation quantization between the Q8 FFN up and gate for 2–8 token batches. | −3–4% step time. |
| `LLAMA_DFLASH_PREFILL_TAIL` | 1 | Every DFlash2 draft layer uses a 2048-token sliding window, so prompt rows more than ~2.1K positions before the prompt end are never attended. They are skipped instead of fed to the drafter, and drafts are unchanged. Adds `common_speculative_set_prompt_end()` and `llama_model_is_swa_all()`. | Fresh 94K prompt 983 → 1,069 t/s. |
| `GGML_SYCL_Q8_DPAS` | off | Our int8 DPAS (XMX) Q8_0 mul_mat for 2–8 tokens. Superseded by upstream #29864, which was 1.8 ms faster. **Don't combine the two.** | 73 → 67 ms when it was the only XMX mat-mul. |

Always on (no switch) in the patched build:
- **SYCL `get_tensor_2d` / `set_tensor_2d`.** Without them, ggml falls back to one synchronous copy per row. Under TP every KV row is split across both GPUs, so a 40 MiB drafter checkpoint took 250 ms (now 31 ms). Agent turns: 804 → 928 t/s prefill.
- **oneDNN fixes:**
  - the GEMM scratchpad is a per-call temporary instead of being cached in the LIFO pool (fixes `GGML_ASSERT(ptr == pool_addr + pool_used)`);
  - the SDPA partition cache is keyed per backend context (fixes "could not execute sdpa primitive" with a second context on the same GPU).

Diagnostic or experimental switches that stay off: `GGML_SYCL_TP_FUSED_AR` (2.2× slower), `GGML_SYCL_BATCH_CPY`, `GGML_SYCL_SHARE_Q8_1`, `GGML_SYCL_Q8_MMVQ_ROWS`, `GGML_SYCL_Q8_GLU_MODE`, `GGML_SYCL_Q8_DPAS_FUSED_GLU`, `GGML_SYCL_FA_GQA6`, `GGML_SYCL_FA_ONEDNN_MIN_Q`, `LLAMA_TEST_TP_SPLIT_TARGET_HEAD`, `LLAMA_DFLASH_PROCESS_TIMING` (logs the drafter prompt pass and checkpoint copy times), and `LLAMA_DFLASH_PROCESS_SKIP` (drafts degrade; for costing only).

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

Prompt processing, agent turns (~5K new tokens at 35–68K):

| | t/s |
|---|---|
| DFlash2 before | 772 |
| + prefill tail | 804 |
| + 2D copies | 928 |
| (no drafter, reference) | 1,020 |

### Drafter choice, measured on real sessions (not benchmarks)

(Single sessions of ~1 hour each, so a smaller sample than the results table at the top.)

Published acceptance (greedy, GSM8K-style) is ~50%+. On real agent work at temperature 1.0 it's ~25–36%, and roughly the same for both drafters: 2.95 tokens per cycle for DFlash2 vs 2.84 for EAGLE-4. DFlash2 wins because it drafts the whole block in one pass, while EAGLE-4 pays per draft token.

| Context | DFlash2 | EAGLE-4 (fork build, rank-128 candidate head) |
|---|---|---|
| 40–80K | 61.8 ms, 50.5 t/s | 70.9 ms, 40.7 t/s |
| 80–120K | 64.9 ms, 45.2 t/s | 71.9 ms, 38.7 t/s |
| 120–160K | 68.3 ms, 40.8 t/s | 78.3 ms, 37.1 t/s |

### Tried, didn't help

- **More than 7 drafts, or token trees:** verify batches over 8 tokens leave the fast XMX kernels.
- **n-gram drafting, probabilistic drafting.**
- **Fused all-reduce kernel:** 2.2× slower.
- **Batched copies, shared activation quantization, multi-row Q8 GEMV:** all correct, none measurably faster. Launch overhead isn't the bottleneck; the GPUs run ahead of the host.
- **Pinning the drafter to one GPU:** crashes, because DFlash2 borrows the target's output head from the cross-GPU buffer.
- **Larger ubatch** (4096/5120): prefill drops to ~480 t/s.
- **Lower temperature** is faster (more drafts accepted), but costs output variety.

## 7. The oneAPI 2026.1 multi-GPU leak (affects any SYCL tensor-parallel setup)

**Symptom:** in long sessions, decode crept from 64 to ~200 ms per cycle, **even after the context was reset**. The server's private memory reached 92 GB on a 32 GB machine, and Windows paged. Benchmarks under ~15 minutes don't show it.

**Cause:** launching the same kernel alternately on two GPUs of one SYCL context leaks host memory on every launch. `repro/oneapi-multigpu-kernel-cache-leak/alt.cpp` reproduces it without llama.cpp:

| 1M launches | Leak per launch |
|---|---|
| Same kernel, one GPU | ~5 B |
| **Alternating GPUs, shared context** | **166 B** |
| Different kernel name per GPU | ~9 B |
| Separate SYCL context per GPU | ~9 B |

`SYCL_CACHE_TRACE=7` shows a fast-cache `Kernel inserted` on every device switch. In the runtime's `kernel_program_cache.hpp`, `saveKernel()` calls `MFastKernelCache.try_emplace(name, FastKernelSubcacheWrapper(...))`. When the key already exists, the temporary wrapper is destroyed, and its destructor erases every entry for the context, including the other GPU's. So every device switch misses. Each miss also appends a `(kernel name, device)` record to `MProgramToFastKernelCacheKeyMap`, which is never pruned. Tensor parallel switches devices thousands of times per decode, which comes to ~0.5 MB per token. Both Level Zero adapters (v1 and v2) are affected, and `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD` doesn't help.

**Workaround** (`GGML_SYCL_PER_DEVICE_CONTEXT=1`): every dpct queue gets its own per-device SYCL context, made with `make_context<level_zero>(native, {device}, ownership::keep)` from the platform default context's native Level Zero context.
- Each fast cache only ever sees one device.
- USM is still shared at the Level Zero level, so host buffers allocated through one GPU's context keep working on the other GPU. `repro/.../shared_ze.cpp` checks that kernels and copies work across them.
- The context-equality checks in the all-reduce become "same native context".

## 8. Verification method

All speed claims compare **time per cycle at temperature 1.0, averaged over 6 fixed seeds** on a 94K-token code prompt. At temperature 1.0, tokens/s for a single sample swings ±25% with the sampled text, while time per cycle is stable to ±0.5 ms. Correctness is checked with a 64-token greedy probe against the layer-split reference (64/64 for every adopted change), three image turns, and checkpoint-restore runs.

## License

MIT (see `LICENSE`). `patches/llama.cpp-d81d0d8fb-dual-b70.patch` modifies [llama.cpp](https://github.com/ggml-org/llama.cpp), which is also MIT-licensed (Copyright (c) 2023-2026 The ggml authors); the code it changes remains under that license.

## Caveats

- One machine, one driver, one model. Other Intel GPUs (Xe, Xe2 iGPU, A-series) are untested.
- The patch targets `d81d0d8fb`; later upstream changes may need rebasing.
- The XMX attention kernel is specialised for D=256 / GQA 6 / F16 KV (Qwen3.8-27B); other shapes fall back to the stock kernels.
