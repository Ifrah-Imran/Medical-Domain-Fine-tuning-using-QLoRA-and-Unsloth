# Medical LLM Fine-tuning using QLoRA and Unsloth

## Objective
To fine-tune a large language model on a medical dataset using QLoRA and Unsloth’s 4-bit quantized training pipeline.

## Tools Used
Python, Google Colab, Unsloth, Transformers, PEFT, PyTorch

## Dataset
- Medical QA dataset (clinical question-answer pairs)
- Domain-specific instruction data

## Methodology
- Loaded base model (Llama 3 / DeepSeek-R1 variant)
- Applied 4-bit quantization using Unsloth
- Configured LoRA adapters for efficient fine-tuning
- Tokenized medical dataset for training
- Trained model using epoch-based QLoRA workflow
- Monitored GPU memory usage in Colab
- Saved trained adapter weights
- Tested model on unseen medical queries

## Workflow
Dataset → Tokenization → 4-bit Quantized Model → LoRA Training → Adapter Saving → Inference Testing

## Results
Successfully adapted a base LLM to a medical domain using efficient low-resource fine-tuning techniques.

## Key Learnings
- QLoRA and PEFT techniques
- 4-bit quantized training efficiency
- Domain adaptation of LLMs
- Memory optimization in large model training
