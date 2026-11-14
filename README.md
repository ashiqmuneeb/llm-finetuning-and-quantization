# 🧬 LLM Fine-Tuning & Quantization

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-green)
![GPU](https://img.shields.io/badge/Runs%20on-Free%20Colab%20T4-F9AB00)

Practical notebooks on adapting and running language models on a **single free GPU**: fine-tuning a summarization model, QLoRA fine-tuning of Llama-2, and running a 13B model with GGUF quantization.

## 📓 Notebooks

| # | Notebook | Topic | Open |
|---|---|---|---|
| 1 | `01_pegasus_samsum_summarization_finetuning` | Fine-tune `google/pegasus-cnn_dailymail` on the SAMSum dialogue dataset (`knkarthick/samsum`) with `Seq2SeqTrainer`, then evaluate with ROUGE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashiqmuneeb/llm-finetuning-and-quantization/blob/main/notebooks/01_pegasus_samsum_summarization_finetuning.ipynb) |
| 2 | `02_llama2_qlora_finetuning` | QLoRA fine-tuning of `NousResearch/Llama-2-7b-chat-hf` on `mlabonne/guanaco-llama2-1k` using 4-bit NF4, LoRA and `SFTTrainer`; merges adapters and can push to the Hub | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashiqmuneeb/llm-finetuning-and-quantization/blob/main/notebooks/02_llama2_qlora_finetuning.ipynb) |
| 3 | `03_llama2_gguf_quantization` | Run `Llama-2-13B-chat` as a 5-bit GGUF (`Q5_K_M`, about 9 GB) with `llama-cpp-python` compiled for CUDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashiqmuneeb/llm-finetuning-and-quantization/blob/main/notebooks/03_llama2_gguf_quantization.ipynb) |

## 🔍 What Each Notebook Covers

### 1. Dialogue summarization with Pegasus
- Loads the SAMSum chat-summary dataset and tokenizes dialogues and summaries
- Trains with `DataCollatorForSeq2Seq` and `Seq2SeqTrainer`
- Evaluates with ROUGE on a small 10-example test subset (ROUGE-1 about 0.35, ROUGE-L about 0.29)
- Saves the fine-tuned model and tokenizer and reloads them for inference

### 2. QLoRA fine-tuning of Llama-2 7B
- Loads the base model in **4-bit (NF4)** with `bitsandbytes`
- Adds **LoRA** adapters on the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Trains with TRL's `SFTTrainer`: 1 epoch, batch size 4, gradient accumulation 2, learning rate 2e-4, cosine schedule, `paged_adamw_8bit`
- Merges the adapters into a standalone model and optionally pushes it to the Hugging Face Hub

### 3. GGUF quantization
- Explains why GGUF replaced the old GGML format
- Downloads a quantized 13B chat model from the Hub and runs it on a T4 with GPU offloading

## ⚙️ Requirements
- A **GPU runtime** (free Colab T4, about 15 GB VRAM, is enough)
- A Hugging Face token for notebook 2 (add it as the Colab secret `HF_TOKEN`)
- Llama-2 weights are subject to Meta's license

## 🚀 How to Run
1. Open a notebook with the **Open in Colab** badge.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Add `HF_TOKEN` in the **🔑 Secrets** panel if prompted.
4. Run the cells in order.

## 🛠️ Tech Stack
Transformers · PEFT · TRL · bitsandbytes · Datasets · Evaluate (ROUGE) · llama-cpp-python

## 📁 Structure
```
llm-finetuning-and-quantization/
├── notebooks/
│   ├── 01_pegasus_samsum_summarization_finetuning.ipynb
│   ├── 02_llama2_qlora_finetuning.ipynb
│   └── 03_llama2_gguf_quantization.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔮 Next Steps
- Evaluate on the full SAMSum test set
- Fine-tune on a custom domain dataset
- Compare 4-bit vs 8-bit quality and speed

## 📄 License
MIT License. See [LICENSE](LICENSE). Model weights follow their own licenses.
