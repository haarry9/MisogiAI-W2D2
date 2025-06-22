# tokenise.py

from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer, pipeline
import json

sentence = "The cat sat on the mat because it was tired."

def run_tokenisation_comparison():
    tokenizers = {
        "BPE (GPT-2)": GPT2Tokenizer.from_pretrained("gpt2"),
        "WordPiece (BERT)": BertTokenizer.from_pretrained("bert-base-uncased"),
        "SentencePiece (T5)": T5Tokenizer.from_pretrained("t5-small")
    }

    compare_md = []

    for name, tokenizer in tokenizers.items():
        encoded = tokenizer.encode_plus(sentence, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        ids = encoded["input_ids"]
        token_count = len(ids)

        compare_md.append(f"## {name}\n")
        compare_md.append(f"**Tokens:**\n`{tokens}`\n")
        compare_md.append(f"**Token IDs:**\n`{ids}`\n")
        compare_md.append(f"**Token Count:** {token_count}\n\n")

    explanation = """
## Why token splits differ across algorithms

Each algorithm uses different principles:
- **BPE (GPT-2)**: Merges most frequent pairs of bytes/subwords. Tokenizes even whitespace and punctuation as separate units.
- **WordPiece (BERT)**: Breaks rare words into subwords prefixed by `##`, ensuring better handling of out-of-vocab terms.
- **SentencePiece (T5)**: Trained directly on raw text without preprocessing (whitespace is treated as a normal symbol), using a probabilistic Unigram model.

These training methods and preprocessing steps cause the token boundaries and granularity to differ.
"""
    compare_md.append(explanation.strip())

    with open("compare.md", "w", encoding="utf-8") as f:
        f.write("\n".join(compare_md))

    print("✅ Tokenization comparison written to compare.md")


def run_mask_predictions():
    masked_sentence = "The <mask> sat on the mat because it was <mask>."
    fill = pipeline("fill-mask", model="roberta-base")

    # First mask prediction
    mask_1 = masked_sentence.replace("<mask>", fill.tokenizer.mask_token, 1)
    mask_1 = mask_1.replace("<mask>", "tired", 1)
    preds_1 = fill(mask_1)

    # Second mask prediction
    mask_2 = masked_sentence.replace("<mask>", "cat", 1)
    mask_2 = mask_2.replace("<mask>", fill.tokenizer.mask_token, 1)
    preds_2 = fill(mask_2)

    results = {
        "mask_1": [pred["token_str"].strip() for pred in preds_1[:3]],
        "mask_2": [pred["token_str"].strip() for pred in preds_2[:3]]
    }

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Fill-mask predictions written to predictions.json")


if __name__ == "__main__":
    run_tokenisation_comparison()
    run_mask_predictions()
