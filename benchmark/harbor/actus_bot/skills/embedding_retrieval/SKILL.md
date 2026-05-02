---
name: embedding_retrieval
description: How to correctly use embedding models for retrieval and ranking, including model-family-specific input formatting (BGE, E5, Instructor, GTE) and asymmetric encoding patterns.
version: 1.1.0
---

# Embedding Retrieval

Procedural guide for using embedding models to encode queries and documents for retrieval and ranking tasks. Different model families have **different input formatting requirements** — using the wrong format silently degrades results.

## When to Use This Skill

- You need to rank or retrieve documents using an embedding model.
- You are writing code that calls `SentenceTransformer`, HuggingFace `AutoModel`, or any embedding API for semantic search.
- You see unexpected retrieval results (e.g., top-ranked documents are not semantically related to the query).
- You are evaluating or comparing embedding models on a retrieval benchmark.

---

## ⚠️ Critical Rules

> **IMPORTANT: Empty `model.prompts` Does NOT Mean "No Prefix Needed"**
>
> If `model.prompts` returns empty strings or an empty dict but the model family (BGE, E5, etc.) is known to require instruction prefixes, you **MUST** use the documented prefix from the model card/README. Do NOT treat empty prompts as evidence that no prefix is needed — it simply means the prompt configuration was not included in this model revision.

> **Decision Hierarchy for Conflicting Information**
>
> When runtime configuration (`model.prompts`, config files) conflicts with official model documentation (README, model card, HuggingFace page), **ALWAYS prefer the official documentation**. The model card is the authoritative source. Hierarchy:
>
> 1. **Model card / README on HuggingFace** — highest authority
> 2. **Published paper** for the model family
> 3. **`config_sentence_transformers.json`** in the model repo
> 4. **`model.prompts`** at runtime — lowest authority (may be empty or stale)

> **Commitment Rule: Do NOT Re-Run Without Prefix to "Compare"**
>
> Once you produce a result using the model card's documented instruction prefix, that is your **final answer**. Do NOT re-run without the prefix to "compare" — this risks overwriting a correct result with an incorrect one. If you want to compare approaches, save results to **separate files** and reason explicitly about which is more trustworthy before writing the final answer.

---

## Step-by-Step Workflow

### Step 1 — Identify the Model Family

Before writing any encoding code, determine which family the model belongs to by inspecting the model name or HuggingFace ID:

| Model ID Pattern | Family | Key Characteristic |
|---|---|---|
| `BAAI/bge-*` | **BGE** | Query instruction prefix required |
| `intfloat/e5-*` or `intfloat/multilingual-e5-*` | **E5** | `"query: "` / `"passage: "` prefixes required |
| `hkunlp/instructor-*` | **Instructor** | Explicit instruction template for both queries and docs |
| `Alibaba-DAMO/gte-*` or `thenlper/gte-*` | **GTE** | Check model card; some variants need prefixes |

**If the model is not in this table**, check its HuggingFace model card and config files before assuming no prefix is needed.

### Step 2 — Inspect Model Configuration

After loading the model, inspect its configuration to find required prompts or instructions.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# Check available prompt names
print(model.prompts)
# Expected output for BGE: {'query': 'Represent this sentence for searching relevant passages: '}
```

> ⚠️ **Remember:** If `model.prompts` returns `{}` or `{'query': ''}`, this does NOT mean no prefix is needed. Cross-reference the model card (see Critical Rules above). The model card is the authoritative source — runtime config can be incomplete.

**Alternative: inspect config files directly.** Look for `config_sentence_transformers.json` in the model's cache directory or on HuggingFace. This file contains prompt definitions under the `"prompts"` key.

```python
import json, os

# Find the cached model directory
cache_dir = model.model_card_data.model_id  # or check ~/.cache/huggingface/hub/
# Read config_sentence_transformers.json for prompt definitions
```

> **Rule:** Never skip this step. Even if you have used a model before, verify — model updates can change prompt requirements.

### Step 3 — Encode with the Correct Pattern

Retrieval models are **asymmetric**: queries and documents must be encoded differently.

#### BGE Models (`BAAI/bge-*`)

**Known instruction prefixes for BGE models (from official model cards):**

| Variant | Query Instruction Prefix |
|---|---|
| **BGE Chinese** (`bge-*-zh-*`) | `为这个句子生成表示以用于检索相关文章：` |
| **BGE English** (`bge-*-en-*`) | `Represent this sentence for searching relevant passages: ` |

These prefixes are **critical** for correct retrieval rankings. Without them, the model treats queries the same as documents and ranking quality collapses.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# Queries: use prompt_name="query" to prepend the instruction
query_embedding = model.encode(
    [query],
    prompt_name="query",
    normalize_embeddings=True
)

# Documents: encode WITHOUT any prompt
doc_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

# Rank by cosine similarity
similarities = query_embedding @ doc_embeddings.T
```

If `prompt_name="query"` is not available (older sentence-transformers version), **or if `model.prompts` is empty**, manually prepend the correct prefix from the table above:

```python
# For BGE English models:
instruction = "Represent this sentence for searching relevant passages: "
query_embedding = model.encode(
    [instruction + query],
    normalize_embeddings=True
)

# For BGE Chinese models (bge-*-zh-*):
instruction_zh = "为这个句子生成表示以用于检索相关文章："
query_embedding = model.encode(
    [instruction_zh + query],
    normalize_embeddings=True
)
```

> ⚠️ **BGE Chinese models specifically:** The prefix `为这个句子生成表示以用于检索相关文章：` uses a Chinese colon (`：`), not an ASCII colon (`:`). Copy-paste carefully.

#### E5 Models (`intfloat/e5-*`)

```python
# Queries: prepend "query: "
query_embedding = model.encode(
    ["query: " + query],
    normalize_embeddings=True
)

# Documents: prepend "passage: "
doc_embeddings = model.encode(
    ["passage: " + doc for doc in documents],
    normalize_embeddings=True
)
```

#### Instructor Models (`hkunlp/instructor-*`)

```python
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large")

# Both queries and documents use instruction templates
query_embedding = model.encode(
    [["Represent the query for retrieving relevant documents: ", query]]
)
doc_embeddings = model.encode(
    [["Represent the document for retrieval: ", doc] for doc in documents]
)
```

#### GTE Models

```python
# Many GTE models need no prefix, but ALWAYS check the model card first.
model = SentenceTransformer("thenlper/gte-base")

query_embedding = model.encode([query], normalize_embeddings=True)
doc_embeddings = model.encode(documents, normalize_embeddings=True)
```

> **Warning for GTE:** Some newer GTE variants (e.g., `gte-Qwen2-*`) do require instruction prefixes. Always verify via `model.prompts` or the model card.

### Step 4 — Compute Similarity and Rank

```python
import numpy as np

# Cosine similarity (embeddings should already be normalized)
similarities = query_embedding @ doc_embeddings.T

# Get ranked indices (highest similarity first)
ranked_indices = np.argsort(-similarities[0])

# Print ranked results
for rank, idx in enumerate(ranked_indices):
    print(f"Rank {rank + 1}: (score={similarities[0][idx]:.4f}) {documents[idx][:80]}")
```

### Step 5 — Validate Results

After computing rankings, perform a **sanity check**:

1. **Semantic coherence**: Do the top-3 results relate to the query topic? If the query is about "machine learning optimization" but the top result is about "cooking recipes", something is wrong.
2. **Score distribution**: If all similarity scores are nearly identical (e.g., all between 0.71 and 0.73), the model may not be discriminating — re-check the encoding approach.
3. **Known-answer test**: If you can identify a document that *should* rank highly, verify it does. If it doesn't, the most likely cause is a missing query prefix.

**If results look wrong**, the most common fix is:
- You forgot the query instruction prefix → go back to Step 3.
- You applied the query prefix to documents (or vice versa) → queries and documents must be encoded differently.
- You forgot to normalize embeddings → add `normalize_embeddings=True`.

> ⚠️ **Commitment Rule reminder:** If you already produced a correct-looking result with the documented prefix, do NOT re-run without it. Stick with the result that used the authoritative prefix.

## Error Handling

| Problem | Likely Cause | Fix |
|---|---|---|
| All similarities are ~equal | Missing query instruction prefix | Add `prompt_name="query"` or manual prefix |
| `model.prompts` is empty `{}` | Prompt config not shipped with this model revision | Use the prefix from the model card (see BGE/E5 tables above) |
| `KeyError: 'query'` on `prompt_name` | Old sentence-transformers version or model lacks prompt config | Manually prepend the instruction string |
| Scores are negative or > 1 | Embeddings not normalized | Add `normalize_embeddings=True` |
| `ImportError` for InstructorEmbedding | Package not installed | `pip install InstructorEmbedding` |
| Unexpected language behavior | Using a language-specific model (e.g., `bge-small-zh`) for the wrong language | Switch to a multilingual or matching-language variant |

## Quick Reference: Decision Checklist

Before writing encoding code, answer these questions:

- [ ] What is the model family? (BGE / E5 / Instructor / GTE / other)
- [ ] Does the model require a query prefix? (Check model card **first**, then `model.prompts`)
- [ ] If `model.prompts` is empty, have I checked the model card for the documented prefix?
- [ ] Does the model require a document prefix? (E5 and Instructor do; BGE does not)
- [ ] Am I normalizing embeddings? (Required for cosine similarity)
- [ ] Have I validated the results with a sanity check?
- [ ] Am I committing to the result that used the authoritative prefix? (No re-running without it)
