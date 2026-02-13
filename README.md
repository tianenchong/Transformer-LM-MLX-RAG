# Transformer LM (MLX) — RAG-enabled

A **research-grade Transformer language model** with **Retrieval-Augmented Generation (RAG)** that answers questions by retrieving document chunks and generating contextual responses. Perfect for showcasing **LLM engineering, RAG pipelines, and domain-specific fine-tuning**.


## Problem Statement

Build a scalable, from-scratch RAG system that:
- **Retrieves** relevant document chunks from a local collection in response to user queries
- **Conditions** a language generator on those chunks to produce accurate, grounded answers
- **Supports** easy fine-tuning on new datasets (Wikipedia, internal knowledge bases, etc.)

The repo demonstrates the full pipeline: data preparation → retriever training → generator training → interactive inference.



## Architecture Diagram

![Architecture](assets/architecture.svg)

**Key components:**

1. **Chunking**: Documents are split into fixed-size windows (tokens)
2. **Chunk Store**: Tokenized chunks indexed for retrieval
3. **Retriever**: Embedding model that maps questions and chunks to a shared space
4. **Top-K Retrieval**: Brute-force inner-product search to rank and select top-K chunks
5. **Generator**: Encoder-Decoder that generates answers conditioned on retrieved chunks
6. **Output**: Generated answer tokens decoded back to text



## Why From-Scratch?

✓ **Full control** over model capacity, tokenizer, and embedding space alignment  
✓ **Reproducible research** on small datasets without external dependencies  
✓ **No licensing or API costs** — everything runs locally on MLX (Apple Silicon optimized)  
✓ **Lightweight** — easy to add, modify, or experiment with components  
✓ **Self-contained** — demonstrates the entire RAG pipeline from scratch in ~1200 lines  


## Embedding Model Used

- **Retriever**: `RecurringTransformerLM` — a small attention-based encoder producing 64-dimensional embeddings
- **Tokenizer**: SentencePiece BPE (`wiki.model`), vocab size 10k
- **Training**: On TriviaQA QA pairs; learns to rank relevant chunks higher


## Chunking Strategy

- Documents are tokenized with SentencePiece and split into **non-overlapping windows**
- Window size = `--context_size` (default 64 tokens)
- Padding applied to fit exact boundaries
- Ensures chunks encode both local context and structural information


## Retrieval Method

**Algorithm**: Brute-force inner-product nearest neighbor search
- Query → embedding via Retriever
- Compute dot-product similarity with all chunk embeddings (scaled by dimension)
- Select top-K using `argpartition`
- **Speed**: ~O(N) for N chunks; no ANN index (future optimization)


## Reranking?

❌ Not currently implemented. **Future enhancement**: Add a cross-encoder reranker to re-score top-K chunks, improving precision at the cost of latency.


## Evaluation Metrics

**Retrieval-side** (To-do):
- **Recall@K**: Fraction of questions where top-K chunks contain the answer
- **MRR**: Mean Reciprocal Rank of the first correct chunk
- **Latency**: End-to-end retrieval + top-K selection time

**Generation-side** (To-do):
- **Exact Match / F1**: Answer token overlap vs. ground truth
- **BLEU / ROUGE**: Sequence-level similarity metrics


# Getting Started

**Recommended Python**: 3.10+

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install packages:

```bash
pip install -r requirements.txt
```

3. (Optional) Create the default model/data directory used by the scripts:

```bash
mkdir -p /Volumes/RAMDisk/transformer-lm-rag
```

## Usage and Workflow

There are three logical steps for using the repository:

1. Prepare datasets and tokenizer (dataset step).
2. Train a retriever model that embeds questions and document chunks (retriever step).
3. Use the retriever to fetch top-K chunks and train an encoder-decoder generator conditioned on those chunks (all/full pipeline).

The `--target` flag controls which step(s) run:

- `--target dataset` — run dataset preparation and tokenizer training.
- `--target retriever` — train only the retriever model.
- `--target all` — run the full RAG pipeline (retrieve top-k chunks for each QA pair, then train the encoder-decoder generator).

### Examples

- Prepare dataset and tokenizer (adjust `--from_dir`/`--to_dir` as needed):

```bash
python main.py --target dataset --from_dir ~/Downloads/triviaqa-rc --to_dir /Volumes/RAMDisk/transformer-lm-rag --save_dir /Volumes/RAMDisk/transformer-lm-rag
```

- Train retriever only (fast example):

```bash
python main.py --target retriever --batch_size 128 --num_iters 10000 --save_dir /Volumes/RAMDisk/transformer-lm-rag
```

- Run the complete RAG pipeline (retrieve + train generator):

```bash
python main.py --target all --batch_size 32 --num_iters 100000 --save_dir /Volumes/RAMDisk/transformer-lm-rag
```

- Interactive inference mode (uses pretrained retriever + generator when available):

```bash
python main.py --target all --inference --save_dir /Volumes/RAMDisk/transformer-lm-rag
```

### Key Flags

- `--save_dir` : path to model/data directory (default: `/Volumes/RAMDisk/transformer-lm-rag`)
- `--from_dir`, `--to_dir` : used by dataset prep to read and write raw input and output directories
- `--target` : `none|dataset|retriever|all` (controls the pipeline stage)
- `--retriever_identifier` : explicit retriever model id to load
- `--identifier` : explicit generator/model id to load
- `--inference` : run interactive generation loop
- `--gpu` : enable Metal backend (if supported)
- `--visualisation` : visualise training progress at a fixed interval
- `--context_size`, `--vocab_size`, `--num_blocks`, `--dim`, `--num_heads`, `--batch_size`, `--num_iters` — model and training hyperparameters


# Technical Details

## File Structure

- [main.py](main.py) — Training and orchestration for retriever and generator
- [datasets.py](datasets.py) — Dataset preparation, tokenizer training, and data loaders
- [requirements.txt](requirements.txt) — Python dependencies
- `assets/architecture.svg` — System architecture diagram

## Data Pipeline Details

- `datasets.prepare_generator_qa_dataset(from_dir, to_dir)` builds simplified QA JSON and expands short answers using an external `mlx_lm` model.
- `datasets.qa_json_to_txt(save_dir)` writes question/answer text files used for tokenizer training.
- `datasets.train_tokenizer(save_dir)` trains a SentencePiece BPE tokenizer at `save_dir/tokenizer/wiki.model` (vocab size 10k).
- `datasets.load_retriever_dataset(window_size, save_dir)` returns chunked document arrays, file metadata, tokenized questions and answers used by the retriever trainer.
- `datasets.load_generator_dataset(save_dir)` returns q/a pairs padded to powers of two and used by the generator training.

## Retriever & Generator

- Retriever: implemented as a small recurring Transformer (`RecurringTransformerLM` in [main.py](main.py)) that maps questions and document chunks to fixed-size embeddings. Training uses contrastive-style losses and samples positive/negative chunks per question.
- Generator: an encoder-decoder-like model uses an encoder for queries and documents and a decoder conditioned on retrieved chunks to generate answers. The script fetches top-K chunks per question using the retriever before training the generator.

## Checkpointing & interrupts

- Checkpoints are written atomically as `<sha1>.<type>.safetensors` and `<sha1>.<type>.metadata` where `<type>` is `retriever` or `generator`.
- Single Ctrl+C: finish current job, save, then exit.
- Double Ctrl+C: force exit immediately (may abort save attempts).

## Requirements & Important Packages

See `requirements.txt` for pinned runtime deps. The codebase uses the following notable packages and tools beyond the standard library:

- `mlx` (MLX core/nn/optimizers)
- `safetensors`
- `dill`
- `sentencepiece` (tokenizer training and tokenization)
- `mlx_lm` (used by [datasets.py](datasets.py) for short-answer expansion)
- `datasets` (dataset helpers used in the code)

## Notes & next steps

- If you rely on a RAM disk, update `--save_dir` accordingly; otherwise choose a local writable path.
- If you want reproducible runs, set `--seed` and use `--new_model` when starting fresh.
- Once the retriever model is finalized, save chunk embeddings to a vector database to avoid redundant computation during future retriever operations. This optimization stores pre-computed embeddings so that the retriever model does not need to re-embed chunks on subsequent queries.

## Future Work

- **Fine-Tuning / Domain Adaptation:**  
  The existing retriever and generator models can be fine-tuned on new datasets.  
  This is **essentially the same training process as used originally**, but with a different dataset:
  - Keep the model architecture and learned weights
  - Replace the training dataset with domain-specific data
  - Run a few training iterations to adapt the model  

  Fine-tuning allows the model to retain its general knowledge while learning new, specific information, e.g., company tribal knowledge.
