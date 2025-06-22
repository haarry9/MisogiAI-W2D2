## BPE (GPT-2)

**Tokens:**
`['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']`

**Token IDs:**
`[464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032, 13]`

**Token Count:** 11


## WordPiece (BERT)

**Tokens:**
`['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.', '[SEP]']`

**Token IDs:**
`[101, 1996, 4937, 2938, 2006, 1996, 13523, 2138, 2009, 2001, 5458, 1012, 102]`

**Token Count:** 13


## SentencePiece (T5)

**Tokens:**
`['▁The', '▁cat', '▁', 's', 'at', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.', '</s>']`

**Token IDs:**
`[37, 1712, 3, 7, 144, 30, 8, 6928, 250, 34, 47, 7718, 5, 1]`

**Token Count:** 14


## Why token splits differ across algorithms

Each algorithm uses different principles:
- **BPE (GPT-2)**: Merges most frequent pairs of bytes/subwords. Tokenizes even whitespace and punctuation as separate units.
- **WordPiece (BERT)**: Breaks rare words into subwords prefixed by `##`, ensuring better handling of out-of-vocab terms.
- **SentencePiece (T5)**: Trained directly on raw text without preprocessing (whitespace is treated as a normal symbol), using a probabilistic Unigram model.

These training methods and preprocessing steps cause the token boundaries and granularity to differ.