# Training-Fine-tuning-GPT-Transformer-for-Medical-Q-A-Conversations

**README GitHub (tiếng Việt)**:

* `GPT_training_from_scratch.ipynb`
* `GPT2_pretrained.ipynb`
* `transformer_training_from_scratch.ipynb`
* `transformer_pretrained.ipynb`
---

## README.md (Vietnamese)

# NLP: Huấn luyện & Fine-tune GPT/Transformer cho hội thoại Hỏi–Đáp Y tế

Repo này gồm 4 notebook triển khai và so sánh **2 hướng tiếp cận**:

1. **Training from scratch (train từ đầu)**
2. **Fine-tuning mô hình pretrained**

Bài toán: mô hình hóa hội thoại theo format:

* **Patient:** `<câu hỏi>`
* **Doctor:** `<câu trả lời>`

Dataset sử dụng: **HealthCareMagic-100k-Chat-Format-en** (HuggingFace) với các cặp hỏi–đáp dạng chat.

---

## 1. Nội dung repo

### A. GPT (Causal LM)

#### 1) `GPT_training_from_scratch.ipynb`

* Train **GPT-like Causal LM từ đầu** (PyTorch tự build).
* Tokenizer: **ByteLevel BPE** (train tokenizer từ dữ liệu).
* Input format:

  * Prompt: `Patient: ...\nDoctor:`
  * Target: phần trả lời của Doctor
* Loss:

  * **CrossEntropyLoss** với `ignore_index=-100`
  * **Mask prompt** để không tính loss trên phần “Patient … Doctor:”
* Optimizer/Scheduler:

  * Adam + **Noam LR schedule**
  * Gradient clipping
  * Early stopping theo validation loss
* Có phần:

  * Training/Validation
  * Test + sinh câu trả lời (generate)
  * Đánh giá BLEU / ROUGE và vẽ biểu đồ so sánh

#### 2) `GPT2_pretrained.ipynb`

* Fine-tune **GPT-2 pretrained** (`MODEL_NAME="gpt2"`) cho nhiệm vụ hội thoại.
* HuggingFace `Trainer` + `TrainingArguments`.
* Dataset class tự xây để:

  * Encode chuỗi `Patient... Doctor... + answer + eos`
  * **Mask prompt** để loss chỉ tập trung vào phần answer
* Có tùy chọn **(Q)LoRA/4bit** (cấu hình bitsandbytes + PEFT), bật/tắt bằng biến cấu hình trong notebook.
* Có phần:

  * Train / resume checkpoint
  * Save model/tokenizer
  * Test + generate

---

### B. Transformer Encoder–Decoder (Seq2Seq)

#### 3) `transformer_training_from_scratch.ipynb`

* Train **Transformer Encoder–Decoder từ đầu** (tự implement bằng PyTorch).
* Tokenizer: **ByteLevel BPE** (tương tự hướng scratch).
* Dataset xây theo cặp:

  * Source: `Patient: <question>`
  * Target: `Doctor: <answer>`
* Mask:

  * `src_key_padding_mask` cho encoder
  * `tgt_mask` gồm padding mask + causal mask cho decoder
* Loss/Optimization:

  * Label smoothing (có class loss riêng)
  * Noam optimizer wrapper
* Có phần:

  * Train/Val
  * Đánh giá Test
  * Sinh câu trả lời

#### 4) `transformer_pretrained.ipynb`

* Fine-tune mô hình Seq2Seq pretrained: **`facebook/bart-base`**
* Sử dụng:

  * `AutoTokenizer`, `AutoModelForSeq2SeqLM`
  * `DataCollatorForSeq2Seq`
  * `Trainer` + `TrainingArguments`
* Pipeline:

  * Tiền xử lý/split dữ liệu
  * Train
  * Evaluate trên test
  * Generate output

---

## 2. Kết quả & đánh giá

Repo có notebook tổng hợp và trực quan hóa các thước đo:

* **Perplexity (PPL)** và **Log-PPL**
* **BLEU**
* **ROUGE**

> Ghi chú: BLEU/ROUGE trong bài toán hội thoại mở thường không phản ánh đầy đủ chất lượng nội dung, nhưng vẫn hữu ích để so sánh tương đối giữa các cấu hình.

---

## 3. Cài đặt môi trường

### Cách nhanh nhất: Google Colab

Các notebook đã có sẵn style chạy Colab (có cell mount Google Drive trong một số file). Bạn chỉ cần upload notebook lên Colab và chạy theo thứ tự.

### Cài local (gợi ý)

Python >= 3.9

Cài các thư viện thường dùng trong notebook:

```bash
pip install torch transformers datasets tokenizers evaluate scikit-learn
pip install accelerate
pip install peft bitsandbytes   # nếu bạn dùng LoRA/4bit
pip install rouge_score         # nếu notebook cần
```

> Nếu chạy GPU NVIDIA trên local, hãy đảm bảo cài PyTorch đúng CUDA.

---

## 4. Hướng dẫn chạy

### A. Training từ đầu

1. Mở `GPT_training_from_scratch.ipynb` hoặc `transformer_training_from_scratch.ipynb`
2. Chạy theo thứ tự:

   * Load dataset
   * Train tokenizer BPE (nếu chưa có)
   * Train model
   * Evaluate + Generate

### B. Fine-tune pretrained

1. Mở `GPT2_pretrained.ipynb` hoặc `transformer_pretrained.ipynb`
2. Chạy:

   * Load tokenizer/model pretrained
   * Chuẩn bị dataset & masking
   * Train (hoặc resume checkpoint)
   * Save model/tokenizer
   * Test + generate

---

## 5. Ý tưởng mở rộng

* Thử các dataset hội thoại khác (đa domain) để tăng tính tổng quát.
* Thêm **instruction format** (system prompt) cho hội thoại y tế.
* Đánh giá nâng cao:

  * BERTScore
  * Human evaluation / rubric
  * Safety filtering (y tế là domain nhạy cảm)

---

## 6. References

* Dataset: `RafaelMPereira/HealthCareMagic-100k-Chat-Format-en` (HuggingFace Datasets)
* Pretrained models:

  * `gpt2` (Causal LM)
  * `facebook/bart-base` (Seq2Seq)

---
