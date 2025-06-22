# Q1 – Tokenization & Fill-in-the-Blank (NLP with Transformers)

This project demonstrates how different tokenization algorithms and a masked language model can be used to:
- Compare tokenization behavior across **BPE**, **WordPiece**, and **SentencePiece**.
- Mask tokens in a sentence and predict them using a fill-mask pipeline.

---

## 🔧 Dependencies

Install the required Python packages:

```bash
pip install transformers sentencepiece
```

### 🚀 How to Run
Execute the script from within the q1 folder:

```python
python tokenise.py
```
This will:
Create a file compare.md containing tokenized outputs and analysis (Part 1).

Create a file predictions.json with top-3 fill-mask predictions for two masked tokens (Part 2).

### 📁 File Structure
```bash
q1/
├── tokenise.py         # Main Python script for both parts
├── compare.md          # Output of tokenization comparison
├── predictions.json    # Output of masked token predictions
└── Readme.md           # Instructions and overview
```

### 📌 Notes
The masked language model used for predictions is roberta-base because models like mistralai/Mistral-7B-Instruct do not support fill-mask tasks (they are causal, not masked).

The tokenizer-model pairings used:

- GPT-2 → BPE

- BERT → WordPiece

- T5 → SentencePiece (Unigram)

### ✍️ Sample Sentence Used
```text
The cat sat on the mat because it was tired.

Masked version for Part 2:
```
```text
The <mask> sat on the mat because it was <mask>.
```

