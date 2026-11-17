from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import sacrebleu
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from tqdm import tqdm


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Local path (or Hugging Face model id) of the base model, e.g.
#   export MODEL_PATH=/path/to/models/<model-dir>
MODEL_PATH = os.environ["MODEL_PATH"]

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

df = pd.read_csv("train.csv")
print(df.head())

dataset = Dataset.from_pandas(df)

def format_example(example):
    return {
        "prompt": [
            {"role": "user", "content": "Translate to ASL gloss:\nEnglish: " + example["text"]}
        ],
        "completion": [
            {"role": "assistant", "content": example["gloss"]}
        ]
    }

dataset = dataset.map(format_example, remove_columns=["text", "gloss"])

dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]
train_valid = dataset["train"]

split = train_valid.train_test_split(test_size=0.125, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print("Train: " + str(len(train_dataset)) + ", Eval: " + str(len(eval_dataset)) + ", Test: " + str(len(test_dataset)))


class GemmaCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        prompts = [
            ex["prompt"][0]["content"] + "\nGloss: "
            for ex in examples
        ]
        completions = [
            ex["completion"][0]["content"] + tokenizer.eos_token
            for ex in examples
        ]

        prompt_tokens = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        full_texts = [p + c for p, c in zip(prompts, completions)]
        batch = self.tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()
        for i, prompt_ids in enumerate(prompt_tokens["input_ids"]):
            labels[i, :len(prompt_ids)] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step: {state.global_step} | Loss: {logs.get('loss', 'N/A')}", flush=True)

training_args = SFTConfig(
    output_dir="./resultsTranslateGemma",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    logging_steps=25,
    save_strategy="no",
    bf16=True,
    report_to="none",
    learning_rate=1e-4,
    remove_unused_columns=False,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_strategy="steps",
    dataset_kwargs={"skip_prepare_dataset": True},
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    args=training_args,
    callbacks=[PrintLossCallback()],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=GemmaCollator(tokenizer, max_length=128),
)

trainer.train()


model.eval()

test_dataset_raw = Dataset.from_pandas(df).train_test_split(
    test_size=0.2, seed=42
)["test"]

test_subset = test_dataset_raw.shuffle(seed=42)

def generate_gloss_batch(texts, batch_size=16):
    tokenizer.padding_side = "left"
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        prompts = ["Translate to ASL gloss:\nEnglish: " + t + "\nGloss:" for t in batch_texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred = pred.split("Translate to ASL gloss:")[0].strip()
            all_preds.append(pred)

    return all_preds

all_texts = [ex["text"] for ex in test_subset]
all_refs = [ex["gloss"] for ex in test_subset]

predictions = generate_gloss_batch(all_texts, batch_size=32)
references = all_refs

rows = []
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

for pred, ref, text in zip(predictions, references, all_texts):
    sent_bleu = sacrebleu.sentence_bleu(pred, [ref]).score
    sent_ter = sacrebleu.sentence_ter(pred, [ref]).score
    scores = rouge.score(ref, pred)

    rows.append({
        "text": text,
        "reference": ref,
        "prediction": pred,
        "sentence_bleu": sent_bleu,
        "sentence_ter": sent_ter,
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    })

bleu1 = BLEU(max_ngram_order=1, effective_order=True, force=True)
bleu2 = BLEU(max_ngram_order=2, effective_order=True, force=True)
bleu3 = BLEU(max_ngram_order=3, effective_order=True, force=True)
bleu4 = BLEU(max_ngram_order=4, effective_order=True, force=True)

print(f"BLEU-1:    {bleu1.corpus_score(predictions, [references]).score:.2f}")
print(f"BLEU-2:    {bleu2.corpus_score(predictions, [references]).score:.2f}")
print(f"BLEU-3:    {bleu3.corpus_score(predictions, [references]).score:.2f}")
print(f"BLEU-4:    {bleu4.corpus_score(predictions, [references]).score:.2f}")

ter = sacrebleu.corpus_ter(predictions, [references])
print(f"TER:       {ter.score:.2f}")

avg_r1 = sum(r["rouge1_f1"] for r in rows) / len(rows)
avg_r2 = sum(r["rouge2_f1"] for r in rows) / len(rows)
avg_rL = sum(r["rougeL_f1"] for r in rows) / len(rows)

print(f"ROUGE-1:   {avg_r1:.4f}")
print(f"ROUGE-2:   {avg_r2:.4f}")
print(f"ROUGE-L:   {avg_rL:.4f}")

rows = sorted(rows, key=lambda x: (x["rougeL_f1"], -x["sentence_ter"], x["sentence_bleu"]))

print("\n--- 10 najgorih primjera ---")
for i, row in enumerate(rows[:10], 1):
    print(f"\n#{i}")
    print(f"ROUGE-L: {row['rougeL_f1']:.4f} | sent-BLEU: {row['sentence_bleu']:.2f} | sent-TER: {row['sentence_ter']:.2f}")
    print("Input:      " + row["text"])
    print("Točno:      " + row["reference"])
    print("Predviđeno: " + row["prediction"])

trainer.save_model("./final_modeltranslate")
tokenizer.save_pretrained("./final_modeltranslate")