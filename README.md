# ASL Gloss Translation with Fine-Tuned LLMs

Translating English sentences into **ASL gloss** by fine-tuning two instruction-tuned LLMs with parameter-efficient fine-tuning (QLoRA), and comparing a general-purpose model against a translation-specialised one.

| Script | Base model |
|---|---|
| `train_gemma.py` | Gemma 4 31B-it |
| `train_translategemma.py` | TranslateGemma 27B-it |

## Task and data

Input is an English sentence and the target is its ASL gloss. Training uses the ASLG-PC12 English–ASL gloss corpus, loaded from a CSV with columns `text` (English) and `gloss` (target). The data is not included in this repo.

## Method

- 4-bit NF4 quantization with double quantization (`bitsandbytes`), bf16 compute
- LoRA via PEFT: `r=32`, `alpha=64`, dropout 0.05, all linear layers
- Training with TRL `SFTTrainer`, prompt tokens masked from the loss (completion-only)
- 5 epochs, per-device batch 16, gradient accumulation 2, LR 1e-4, cosine schedule, 100 warmup steps, 8-bit paged AdamW, max sequence length 128
- Prompt format: `Translate to ASL gloss:\nEnglish: <sentence>\nGloss:`
- Greedy decoding (`max_new_tokens=40`, `repetition_penalty=1.3`)

**Data split** (seed 42): 80% / 20% train / test, then the training part is split again 87.5% / 12.5% into train / validation, giving **70% train, 10% validation, 20% test**.

## Results (test set, `train.csv`)

Same data, same split (seed 42), same hyperparameters; test set of 17,542 sentences.

| Model | BLEU-4 | TER (lower is better) |
|---|---|---|
| Gemma 4 31B | **98.74** | **0.98** |
| TranslateGemma 27B | 97.65 | 1.37 |

Repeated runs differ slightly (Gemma 98.74 to 98.76 BLEU-4, TranslateGemma 97.46 to 97.65), so differences of a few tenths of a point are within run-to-run variation; the gap between the two models is larger than that.

### Notes

- **The two scripts differ only in the base model** and minor implementation details (the token-type field each model's collator requires, logging interval, and inference batch size).
- **The reference glosses are rule-generated and noisy.** The "10 worst examples" printout shows misaligned or truncated references (e.g. a one-word gloss for a 15-word sentence) and non-English or badly encoded input. Removing such rows did not improve TranslateGemma (BLEU-4 97.53 on a cleaned subset, which has a different test set and so is not directly comparable).
- Near-saturated scores are expected on this benchmark: the model learns the gloss conversion rules almost perfectly. They should not be read as general ASL translation quality.

## Usage

```bash
pip install -r requirements.txt
export MODEL_PATH=/path/to/base/model        # local directory with the model weights
export HF_TOKEN=...                          # optional, only if you need to log in
# place train.csv with columns: text, gloss
python train_gemma.py                        # or train_translategemma.py
```

The scripts train, evaluate on the held-out test split (BLEU-1..4, TER, ROUGE-1/2/L), print the 10 worst examples, and save the adapter and tokenizer to `final_model*`. Training was run on a GPU cluster and needs a large-memory GPU.
