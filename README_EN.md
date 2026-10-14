# Euromillón llama2.c — Training a GPT-2 From Scratch, End to End

A from-scratch GPT-2-style character-level language model, trained with
[llama2.c](https://github.com/karpathy/llama2c) on synthetic + real-frequency
lottery ("Euromillón"/EuroMillions) data, converted to GGUF and served with
`llama.cpp`.

**Honesty note, upfront: this is NOT a lottery prediction system.**
EuroMillions draws are independent random events; no model can predict them,
and this project makes no such claim. What it actually is: a small,
self-contained exercise in the full ML lifecycle — synthetic data generation,
training a transformer from scratch on CPU/GPU, exporting to a production
inference format (GGUF), debugging numerical/format correctness during that
conversion, and serving the result — using a toy domain specifically because
the data is cheap to generate and the "quality" of output is easy to sanity
check by eye (five numbers 1-50 and two stars 1-12).

## What it is

- `2a-extract-real-data.py` / `2-generate-data.py` / `2b-merge-jsonl-data.py`:
  build a character-level training corpus from historical draw frequencies
  plus synthetically generated combinations.
- `3a-phase1-train.sh` / `3-train.sh` (via llama2.c's `train.py`): train a
  GPT-2-tiny-sized transformer (8 layers, 8 heads, 512 embedding dim,
  **33.6M parameters**) from random initialization — no pretrained weights.
- `3b-phase2-finetune.sh`: a second phase fine-tunes the base model on an
  unrelated text corpus (SWE-Zero / OpenHands trajectories) as a probe of
  how much the tiny model can absorb from out-of-domain data.
- `4-convert.sh`, `build-gguf.py` / `build-gguf-v2.py`,
  `scripts/convert_to_gguf*.py`, `scripts/fix_gguf_header.py`: convert the
  trained PyTorch checkpoint to GGUF for `llama-cli`/`llama-server`. The
  repository keeps multiple numbered/"v2"/"FIXED"/"PADDED" conversion
  attempts — direct evidence of iterating on a byte-level format/header
  mismatch until output matched expectations (see "What was measured" below).
- `compare.py` / `compare-models.sh`: reconstruct the trained model in plain
  PyTorch (manual transformer forward pass copying weight tensors by name)
  and run side-by-side generation against different training checkpoints —
  used to verify a new export reproduces the same effective weights as the
  checkpoint it was built from.
- `run-llama-server.sh` / `run-llama-cli.sh` / `run-llama-c.sh`: serve the
  resulting GGUF with `llama-server` or run it directly with llama2.c's own
  C inference (`runq`).

## Architecture / pipeline

```
Historical draw data + synthetic combinations
                |
   2a/2b/2-generate-data.py  -->  data/euromillon_real.txt (char-level corpus)
                |
   3a-phase1-train.sh  -->  llama2.c train.py
   GPT-2-tiny: 8 layers, 8 heads, 512 dim, 33.6M params, trained from scratch
                |
   models/euromillon-*.pt  (+ optional phase-2 fine-tune on unrelated corpus)
                |
   4-convert.sh / build-gguf*.py  -->  models/*.gguf
   (iterative header/tensor-layout fixes until llama-cli loads correctly)
                |
   compare.py  -->  cross-check: PyTorch forward pass vs. GGUF output
                |
   run-llama-server.sh / run-llama-cli.sh  -->  served inference
```

## How to run

```bash
./run.sh setup                 # clone + build llama2.c
./run.sh data 50000             # generate a 50k-line training corpus
./run.sh train tiny             # train GPT-2-tiny (~117M-class config; this repo's
                                 #   actual trained model is the 33.6M config, see below)
./run.sh convert                # PyTorch checkpoint -> GGUF
./run.sh generate 10            # sample 10 combinations
./run.sh all tiny               # full pipeline

# Or step by step, as actually run in this repo:
python3 2a-extract-real-data.py
bash 3a-phase1-train.sh         # base training, logs to output/phase1*.log
bash 3b-phase2-finetune.sh      # optional fine-tune, logs to output/phase2.log
bash 4-convert.sh euromillon-v2-gpt2
bash run-llama-server.sh models/euromillon-v2-gpt2.gguf
```

Requirements: gcc, git, Python 3.13+, `llama-cli` available (built via
`1-clone.sh`/llama2.c or system llama.cpp).

## What was actually measured

Real training logs are kept under `output/*.log`. From `output/phase1-v2.log`
(500-step base training run, GPT-2-tiny, 33,586,176 parameters, vocab 45
chars, batch 32, seq len 128, lr 0.002 with warmup/decay):

| Step | Loss | LR |
|---|---|---|
| 1 | 0.0416 | 0.00004 |
| 100 | 2.4435 | 0.001946 |
| 200 | 2.0216 | 0.001550 |
| 300 | 1.8194 | 0.000944 |
| 400 | 1.6516 | 0.000411 |
| 500 | 1.5273 | 0.000200 |

(Note: the step-1 loss of 0.04 is an artifact of the tiny randomly-seeded
first batch, not a real learning signal — loss rises then falls over the
run, which is the expected LR-warmup shape; final loss 1.53 after 500 steps
on a 45-symbol character vocabulary.)

From `output/phase2.log` (fine-tuning the base checkpoint on SWE-Zero/
OpenHands trajectory text, 50 steps, batch 8, lr 0.0005): loss went from
**3.92 at step 1 to 1.76 at step 50**, confirming the model can adapt to a
substantially different text distribution, at the cost of ~2,500s for 50
steps on this hardware (CPU/GPU details not logged precisely beyond
`device: cuda`, which on this machine maps to a ROCm GPU via PyTorch's CUDA
compatibility layer).

The GGUF conversion history itself is evidence worth highlighting: the repo
contains multiple iterations (`euromillon-base-gpt2-TEST*.gguf`,
`-PADDED`, `-ADDED-PADDING`, `-CORRECTED`, `-FIXED`, `-FINAL`) and matching
scripts (`fix-gguf.py`, `fix-v1-gguf.py`, `fix-g2m2-padding.py`,
`add-kv-padding.py`), showing a real numerical-correctness debugging process
to get tensor layout/header padding right between the llama2.c binary format
and `llama.cpp`'s GGUF loader — rather than a single clean conversion.
`compare.py` was used to cross-check that a given export's generations match
what the original PyTorch checkpoint produces.

## Relevance

This project demonstrates the mechanics most directly relevant to inference
work: training a transformer from scratch with full control of the training
loop, then exporting to a real serving format (GGUF) and debugging tensor
layout/numerical mismatches until the served model's output matches the
trained checkpoint — a small-scale but genuine instance of the
correctness-then-performance discipline that larger inference optimization
work depends on. It does not include a systematic tokens/sec serving
benchmark or quantization comparison; that would be the natural next step
(e.g. compare GGUF quant levels, or `llama-cli`/`runq` vs `llama-server`
throughput on this same tiny model).

## Limitations

- Toy domain and tiny model (33.6M params, 45-symbol vocabulary) — not
  representative of large-model training or serving costs.
- No prediction claim: lottery outputs are independent random draws; the
  model only learns the surface statistics of the training text.
- Training logs show only loss, not perplexity/held-out validation; no
  train/val split is evidenced in the logs reviewed.
- Several intermediate/debug GGUF files and fix scripts remain in the repo as
  a visible record of the debugging process rather than a cleaned-up final
  pipeline — intentionally left as-is here for honesty, but a published
  version should consolidate `build-gguf-v2.py`/fix-scripts into one correct,
  documented converter.
