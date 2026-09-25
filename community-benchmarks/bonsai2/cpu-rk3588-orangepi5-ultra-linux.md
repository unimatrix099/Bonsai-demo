# Orange Pi 5 Ultra (RK3588) - CPU / Linux

## Summary

Bonsai 2 27B on an Orange Pi 5 Ultra (Rockchip RK3588, 16 GB LPDDR, no discrete
GPU / no usable Vulkan), CPU-only inference, using release `prism-b10735-842b188`
(prebuilt `ubuntu-arm64` asset). This is a low-power ARM SBC data point; expect
throughput far below the GPU reports in this folder.

| Format | PP512 (t/s) | TG128 (t/s) |
|--------|------------:|------------:|
| PQ2_0 | 0.40 | 0.32 |
| PTQ1_0 | 0.49 | 0.38 |
| Q2_0 (development) | not tested | not tested |

Both bands produce correct output (see observations); the fork's Hadamard /
sign-flip transforms work on the ARM CPU backend. On this CPU **PTQ1_0 is both
smaller and faster than PQ2_0** (~19% higher tg, ~23% higher pp).

## Configuration

- Model repository, exact filename(s), and revision: `prism-ml/Ternary-Bonsai-2-27B-gguf`
  (main branch), files `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GiB, reported
  `PQ2_0 - 2.13 bpw (group 128)`) and `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
  (5.53 GiB, reported `PTQ1_0 - 1.75 bpw ternary (group 128)`).
- llama.cpp release/commit and downloaded asset: `prism-b10735-842b188`,
  asset `llama-prism-b10735-842b188-bin-ubuntu-arm64.tar.gz` (CPU build). At runtime
  it auto-loads `libggml-cpu-armv8.2_2.so` (the armv8.2 + dotprod variant). Not built
  from source.
- OS / backend: `Linux orangepi5-ultra 6.1.115-vendor-rk35xx aarch64`; ggml CPU backend
  (RPC backend also loaded but unused). No GPU offload; the board has no libvulkan.
- CPU / RAM: RK3588 — 4× Cortex-A76 (measured ~2.35 GHz under load) + 4× Cortex-A55,
  8 cores total; 16 GB LPDDR. ARMv8.2 with dotprod; no i8mm/bf16/SVE.
- Offload / flash attention / KV / threads / batch: `-ngl 0` (CPU), `-fa on`, `-t 8`,
  `-p 512 -n 128`, `-r 2`, default KV types. `ulimit -n 65536` before running.
- Tuning: CPU governor set to `performance` for the runs (reset to `schedutil`
  afterwards). SoC ~48 °C at load, no thermal throttling observed.

## llama-bench results

Not run via `./setup.sh` — the prebuilt `ubuntu-arm64` CPU asset and both GGUF
files were downloaded directly, since the board runs headless CPU-only. Command
(run from the extracted binary directory with `LD_LIBRARY_PATH` set to it):

```bash
BENCH=./llama-bench   # from llama-prism-b10735-842b188-bin-ubuntu-arm64
"$BENCH" \
  -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 0 -fa on -t 8 -p 512 -n 128 -r 2
```

Raw output:

```
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CPU        |       8 |   1 |           pp512 |          0.40 ± 0.00 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CPU        |       8 |   1 |           tg128 |          0.32 ± 0.00 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CPU        |       8 |   1 |           pp512 |          0.49 ± 0.00 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CPU        |       8 |   1 |           tg128 |          0.38 ± 0.00 |

build: 842b18804 (10735)
```

Note: `-fa` on this build takes `on|off|auto` (not `1`); an invalid `-fa` value makes
`llama-bench` exit without printing a table.

### Q2_0 (development)

Not tested.

## Additional observations

- **Correctness check (separate from llama-bench).** `llama-cli` on PQ2_0
  (`-ngl 0 -t 8 -st -rea off`) answered "The capital of France is Paris." — coherent,
  not gibberish — confirming the fork's Hadamard/sign-flip transforms run correctly on
  the ARM CPU backend. (Stock llama.cpp recognizes `qwen35`/`Q2_0` but omits those
  transforms and would produce incorrect output.)
- **PTQ1_0 > PQ2_0 on this CPU.** PTQ1_0 is ~19% faster at tg and ~23% faster at pp,
  and 1.2 GiB smaller — the smaller footprint outweighs PQ2_0's native ARM dot-product
  kernel here. PTQ1_0 is the recommended band for this board.
- **Throughput context.** ~0.3–0.4 tg is "submit and wait", not interactive, on this
  SBC. The model is dense (all 27 B params active per token) and applies per-token
  Hadamard rotation, so it is much heavier than a similarly-sized sparse-MoE or plain
  Q4_0 model on the same CPU. Do not infer correctness or long-context/vision behavior
  from these pp512/tg128 numbers; vision and long-context were not tested.

- **Custom-kernel experiment (not the stock build).** A `perf` profile of PTQ1_0
  decode showed 86% of time in `ggml_vec_dot_ptq1_0_q8_0`, which had only a scalar
  path on ARM. Adding a NEON version of that kernel (base-3 unpack + SDOT, bit-exact
  with the generic) raised PTQ1_0 tg from 0.38 to 0.67 t/s (1.76x) on this board.
  This is a local fork change, not the released binary; the kernel and details are in
  the llama.cpp fork branch `opt/ptq1_0-arm-neon`
  (`docs/development/rk3588-ptq1_0-neon.md`). Native `-mcpu` flags and thread/affinity
  tuning gave no gain on their own.
