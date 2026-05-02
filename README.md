<div align="center">

<br />

<h1>
  ⚡ jax-rag-bench
</h1>

<h3>
  A reproducible benchmark harness for Retrieval-Augmented Generation —<br/>
  fast embeddings on JAX, ANN search via ScaNN, ranking metrics that don't lie.
</h3>

<br />

<p>
<a href="https://github.com/vkohli123/jax-rag-bench/actions"><img src="https://img.shields.io/github/actions/workflow/status/vkohli123/jax-rag-bench/ci.yml?branch=main&style=flat-square&label=ci&logo=github" alt="CI" /></a>
<a href="https://pypi.org/project/jax-rag-bench/"><img src="https://img.shields.io/pypi/v/jax-rag-bench?style=flat-square&color=blue&logo=pypi&logoColor=white" alt="PyPI" /></a>
<a href="https://pypi.org/project/jax-rag-bench/"><img src="https://img.shields.io/pypi/pyversions/jax-rag-bench?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
<a href="https://github.com/vkohli123/jax-rag-bench/blob/main/LICENSE"><img src="https://img.shields.io/github/license/vkohli123/jax-rag-bench?style=flat-square&color=green" alt="License" /></a>
<a href="https://github.com/vkohli123/jax-rag-bench/stargazers"><img src="https://img.shields.io/github/stars/vkohli123/jax-rag-bench?style=flat-square&color=yellow&logo=github" alt="Stars" /></a>
<a href="https://github.com/vkohli123/jax-rag-bench/discussions"><img src="https://img.shields.io/github/discussions/vkohli123/jax-rag-bench?style=flat-square&color=orange&logo=github" alt="Discussions" /></a>
<p>
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

<br />

<sub><b>Status:</b> active development · <b>Maintainer:</b> <a href="https://github.com/vkohli123">@vkohli123</a> · <b>License:</b> Apache-2.0</sub>

</div>

<br />

---

## ✨ Why jax-rag-bench?

Modern RAG stacks are full of moving pieces — embedding models, chunking strategies, ANN indices, rerankers, hybrid scoring, query rewriting. **Every blog post claims state-of-the-art on a different dataset, with a different recall@k, on a different GPU.** It's chaos.

`jax-rag-bench` is a small, opinionated harness that answers one question well:

> *"Given **this** corpus, **this** query set, and **this** retriever — what's my actual nDCG@10, and how fast does it run?"*

No notebooks held together with duct tape. No `pip install` that drags in 4 GB of CUDA. No metric definitions that quietly disagree with the literature. Just a clean pipeline you can point at any corpus, swap any encoder, and get publication-grade numbers in a single command.

<br />

## 🚀 Features

- **🧠 Pluggable encoders** — `sentence-transformers`, OpenAI, Cohere, or any callable that returns vectors. Swap with one line.
- **⚡ JAX-accelerated embeddings** — drop-in `jax` backend for SentenceTransformer-style models. ~3× throughput vs. eager PyTorch on a single A100; near-linear scaling on TPU v5e pods.
- **🔍 ScaNN under the hood** — Google's anisotropic vector quantizer, the fastest open-source ANN library on most billion-scale corpora. Tuned defaults that don't require you to read the paper.
- **📊 Honest evaluation** — nDCG@k, Recall@k, MRR, MAP, and Hit@k computed per the BEIR/TREC reference implementations. We diff our outputs against `pytrec_eval` in CI.
- **🎛 Reproducible by design** — every run produces a `run.yaml` with the exact corpus hash, encoder revision, ScaNN config, and metric values. Re-run a result from six months ago with one command.
- **📈 Built-in datasets** — first-class support for BEIR, MS MARCO, NQ, HotpotQA, FiQA, and SciFact. Bring-your-own-corpus is a 5-line subclass.
- **🪶 Lightweight** — pure Python, no Docker required, < 200 MB install footprint without GPU extras.
- **🧪 Battle-tested** — 94% line coverage, deterministic seeds, golden-output regression tests on every PR.

<br />

## 📦 Installation

### From PyPI

```bash
pip install jax-rag-bench
```

### With GPU / TPU acceleration

```bash
# CUDA 12
pip install "jax-rag-bench[gpu]"

# TPU (Cloud TPU VMs)
pip install "jax-rag-bench[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### From source

```bash
git clone https://github.com/vkohli123/jax-rag-bench.git
cd jax-rag-bench
pip install -e ".[dev]"
pre-commit install
```

### Verify

```bash
jrb --version
jrb doctor   # checks JAX devices, ScaNN install, and dataset cache
```

<br />

## ⚡ Quickstart

Benchmark `all-MiniLM-L6-v2` on the SciFact dataset in **under 90 seconds**:

```bash
jrb run \
  --dataset scifact \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --index scann \
  --metrics ndcg@10,recall@100,mrr@10
```

Output:

```
┌─────────────────────────────────────────────────────────────┐
│  jax-rag-bench · scifact · all-MiniLM-L6-v2                 │
├─────────────────────────────────────────────────────────────┤
│  corpus size            5,183 docs                          │
│  query set                300 queries                       │
│  embedding throughput  4,210 docs/s   (jax/A100)            │
│  index build              1.2 s                             │
│  query latency p50      0.41 ms                             │
│  query latency p99      1.18 ms                             │
├─────────────────────────────────────────────────────────────┤
│  nDCG@10                 0.6841                             │
│  Recall@100              0.9233                             │
│  MRR@10                  0.6519                             │
└─────────────────────────────────────────────────────────────┘
✓ run saved to runs/2026-05-02T11-04-22Z_scifact_minilm.yaml
```

That's it. No config files, no pickle archaeology.

<br />

## 🏗 Architecture

```
                                      ┌───────────────────┐
        ┌────────────┐                 │  Encoder Backend  │
        │   Corpus   │ ──── chunks ──▶ │   ┌─────────────┐ │
        │  (jsonl /  │                 │   │   JAX       │ │
        │   parquet) │                 │   │ SentenceTx  │ │
        └────────────┘                 │   │  OpenAI…    │ │
                                       │   └──────┬──────┘ │
                                       └──────────┼────────┘
                                                  │ vectors (float16/32)
                                                  ▼
                                       ┌───────────────────┐
                                       │   ScaNN Index     │
                                       │  ┌─────────────┐  │
                                       │  │  AH + Tree  │  │
                                       │  │  Reorder    │  │
                                       │  └─────────────┘  │
                                       └─────────┬─────────┘
                                                 │ top-k
        ┌────────────┐                           ▼
        │  Queries   │ ──── encode ───▶ ┌───────────────────┐
        │  + qrels   │                  │     Evaluator     │
        └─────┬──────┘                  │  nDCG · Recall    │
              │                         │  MRR  · MAP       │
              └────────── qrels ──────▶ │  Hit@k            │
                                        └─────────┬─────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │   run.yaml   │
                                          │  + metrics   │
                                          │  + traces    │
                                          └──────────────┘
```

Five components, each replaceable:

| Component | Default | Alternatives |
|-----------|---------|--------------|
| `Corpus` | `JsonlCorpus` | `ParquetCorpus`, `BEIRCorpus`, `HuggingFaceCorpus` |
| `Encoder` | `JaxSentenceTransformer` | `TorchSentenceTransformer`, `OpenAIEncoder`, `CohereEncoder`, `BYOEncoder` |
| `Index` | `ScaNNIndex` | `FaissIndex`, `HNSWIndex`, `BruteForceIndex` |
| `Evaluator` | `BEIREvaluator` | `TRECEvaluator`, `CustomEvaluator` |
| `Reporter` | `RichConsoleReporter` | `WandbReporter`, `MLFlowReporter`, `JSONReporter` |

<br />

## 📊 Benchmarks

Reproduced on `n1-standard-16` + `1× A100 40GB`, ScaNN 1.3.x, JAX 0.4.x, batch size 256.

### Retrieval quality on BEIR (nDCG@10, higher is better)

| Encoder | SciFact | NFCorpus | FiQA | TREC-COVID | NQ |
|---------|--------:|---------:|-----:|-----------:|---:|
| `all-MiniLM-L6-v2`              | 0.684 | 0.317 | 0.336 | 0.547 | 0.479 |
| `all-mpnet-base-v2`             | 0.711 | 0.331 | 0.382 | 0.681 | 0.531 |
| `bge-small-en-v1.5`             | 0.712 | 0.342 | 0.401 | 0.701 | 0.541 |
| `bge-large-en-v1.5`             | **0.746** | **0.359** | **0.448** | **0.745** | **0.563** |
| `e5-base-v2`                    | 0.731 | 0.348 | 0.412 | 0.722 | 0.553 |

### Throughput (documents / second, single device)

| Backend | MiniLM-L6 | MPNet-base | BGE-large |
|---------|----------:|-----------:|----------:|
| PyTorch (eager)        | 1,420 | 612 | 178 |
| PyTorch (`torch.compile`) | 2,310 | 1,054 | 312 |
| **JAX (this repo)**       | **4,210** | **1,872** | **541** |
| JAX + TPU v5e             | 6,890 | 3,140 | 982 |

> Numbers are reproducible via `make benchmark`. We commit raw run logs under [`benchmarks/results/`](./benchmarks/results) so you can audit every claim.

### Index trade-offs (SciFact, BGE-large)

| Index | Build time | Latency p50 | Recall@100 vs. brute force |
|-------|-----------:|------------:|---------------------------:|
| BruteForce  | 0.0 s | 4.2 ms | 1.000 |
| FAISS-HNSW  | 8.1 s | 0.9 ms | 0.987 |
| **ScaNN**   | **1.2 s** | **0.4 ms** | **0.994** |

<br />

## 🛠 Usage

### Programmatic API

```python
from jrb import Corpus, JaxEncoder, ScaNNIndex, Evaluator

# 1. Load (or stream) your corpus
corpus = Corpus.from_jsonl("data/my_docs.jsonl", text_field="content")

# 2. Pick an encoder — anything that produces (n, d) vectors
encoder = JaxEncoder("BAAI/bge-base-en-v1.5", precision="bf16")

# 3. Embed and index
embeddings = encoder.encode(corpus.texts(), batch_size=256, show_progress=True)
index = ScaNNIndex.build(
    embeddings,
    num_leaves=2000,
    num_leaves_to_search=100,
    reorder=200,
)

# 4. Query
hits = index.search(encoder.encode(["What is anisotropic quantization?"]), k=10)

# 5. Evaluate against qrels
evaluator = Evaluator(qrels="data/qrels.tsv", metrics=["ndcg@10", "recall@100"])
report = evaluator.run(index, encoder, queries="data/queries.tsv")
print(report)
```

### CLI reference

```bash
# Run a single benchmark
jrb run --dataset <name> --encoder <hf-id> --index <scann|faiss|hnsw|bf>

# Sweep multiple encoders
jrb sweep --config configs/sweep_beir_small.yaml

# Compare runs side-by-side
jrb compare runs/run_a.yaml runs/run_b.yaml

# Export to LaTeX / Markdown for your paper
jrb export runs/*.yaml --format latex > tables/retrieval.tex

# Inspect cached datasets
jrb dataset list
jrb dataset info scifact
```

### Bring your own corpus

Subclass `Corpus` and implement two methods:

```python
from jrb import Corpus

class MyCorpus(Corpus):
    def __iter__(self):
        for row in my_database.iterate():
            yield {"id": row.id, "text": row.body, "title": row.title}

    def __len__(self):
        return my_database.count()
```

That's the whole interface. Streaming is supported by default — no need to fit your corpus in RAM.

### Bring your own encoder

```python
from jrb import BaseEncoder
import numpy as np

class MyEncoder(BaseEncoder):
    dim = 768

    def encode(self, texts: list[str], **kw) -> np.ndarray:
        return my_model.embed(texts)  # returns (n, 768) float32
```

Plug it into any pipeline. The harness handles batching, normalization, dtype casting, and progress bars.

<br />

## ⚙️ Configuration

For repeatable experiments, drive everything from a YAML file:

```yaml
# configs/my_experiment.yaml
seed: 1337

dataset:
  name: scifact
  split: test

encoder:
  type: jax_sentence_transformer
  model: BAAI/bge-large-en-v1.5
  precision: bf16
  batch_size: 128
  normalize: true

index:
  type: scann
  num_leaves: 2000
  num_leaves_to_search: 100
  reorder: 200
  distance: dot_product

evaluation:
  metrics: [ndcg@10, recall@100, mrr@10]
  cutoffs: [1, 3, 5, 10, 100]

reporter:
  console: true
  wandb:
    project: rag-bench
    tags: [bge, scifact]
```

```bash
jrb run --config configs/my_experiment.yaml
```

<br />

## 📚 Supported Datasets

| Dataset | Domain | Queries | Corpus | Source |
|---------|--------|--------:|-------:|--------|
| SciFact      | Scientific claim verification | 300 | 5K | BEIR |
| NFCorpus     | Medical IR | 323 | 3.6K | BEIR |
| FiQA         | Financial QA | 648 | 57K | BEIR |
| TREC-COVID   | Biomedical | 50 | 171K | BEIR |
| NQ           | Open-domain QA | 3,452 | 2.6M | BEIR |
| HotpotQA     | Multi-hop QA | 7,405 | 5.2M | BEIR |
| MS MARCO     | Web passage ranking | 6,980 | 8.8M | TREC |

Adding a new dataset is a 30-line subclass of `Dataset`. See [`docs/datasets.md`](./docs/datasets.md).

<br />

## 📐 Metrics, Defined

We get this right because the literature often does not. All metrics are computed per-query and averaged uniformly across the query set, matching `pytrec_eval`.

- **nDCG@k** — graded relevance, log-discounted. The single best metric for ranked retrieval. Default cutoff: 10.
- **Recall@k** — fraction of relevant documents in the top-k. Useful when downstream re-rankers will resort.
- **MRR@k** — reciprocal rank of the first relevant hit. Strict, sensitive to the top result.
- **MAP** — mean average precision over relevant documents. Old-school, still informative.
- **Hit@k** — binary: was *any* relevant doc retrieved in top-k? The weakest signal but easy to communicate.

We provide a [metrics cheatsheet](./docs/metrics.md) explaining when each one misleads you.

<br />

## 🗺 Roadmap

- [x] BEIR datasets, ScaNN, JAX encoder backend
- [x] CLI + programmatic API
- [x] Reproducibility manifests (`run.yaml`)
- [x] W&B reporter
- [ ] **Q3 2026** — hybrid retrieval (BM25 + dense fusion via RRF / weighted)
- [ ] **Q3 2026** — cross-encoder reranking stage
- [ ] **Q4 2026** — built-in support for ColBERT / late-interaction
- [ ] **Q4 2026** — distributed indexing for billion-scale corpora
- [ ] **2027** — multi-vector and matryoshka embedding evaluation
- [ ] **2027** — LLM-as-judge for end-to-end RAG evaluation

Vote on issues with 👍 — that's how we prioritize.

<br />

## ❓ FAQ

<details>
<summary><b>Does this require a GPU?</b></summary>

No. Everything runs on CPU; you'll just embed slower. The PyPI install is CPU-only by default. Add `[gpu]` or `[tpu]` extras when you want acceleration.
</details>

<details>
<summary><b>Why ScaNN and not FAISS?</b></summary>

Both are supported. ScaNN is the default because anisotropic quantization gives better recall-vs-latency curves on most embedding distributions in our tests, and the build times are 5–10× faster. FAISS-HNSW is a perfectly reasonable choice for many workloads — `jrb run --index hnsw` and you're off.
</details>

<details>
<summary><b>How is this different from BEIR / mteb?</b></summary>

[BEIR](https://github.com/beir-cellar/beir) is a dataset collection plus a runner. [MTEB](https://github.com/embeddings-benchmark/mteb) is a leaderboard. `jax-rag-bench` is a *harness* — it consumes BEIR-style datasets and produces results in their format, but it's optimized for fast iteration on your own corpus, your own encoder, and your own ANN config. Think of it as the local development tool you reach for *before* you submit to MTEB.
</details>

<details>
<summary><b>Can I evaluate end-to-end RAG (retrieval + generation)?</b></summary>

Not yet — see roadmap. This repo is intentionally scoped to retrieval. Generation evaluation requires a different machinery (LLM-as-judge, faithfulness, answer correctness) which we'll add in 2027.
</details>

<details>
<summary><b>Does it support reranking?</b></summary>

Cross-encoder reranking ships in Q3 2026. For now you can manually rerank `index.search()` outputs with any model.
</details>

<details>
<summary><b>How are metrics validated?</b></summary>

Every PR runs a CI job that diffs our metric outputs against `pytrec_eval` on a fixed run. If the values disagree by more than 1e-6, the build fails.
</details>

<br />

## 🤝 Contributing

Contributions are warmly welcomed. The bar for a PR is low — even a typo fix or a benchmark on a new dataset is genuinely valuable.

1. Fork the repo and create a feature branch
2. `pip install -e ".[dev]"` and `pre-commit install`
3. Write a test (we use `pytest`)
4. `make lint && make test` should be green
5. Open a PR — describe the *why*, not just the *what*

Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full guide and [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) for the community norms.

### Good first issues

Look for the [`good first issue`](https://github.com/vkohli123/jax-rag-bench/labels/good%20first%20issue) label. Things like adding a dataset loader, writing a docs example, or porting a metric are perfect entry points.

<br />

## 📖 Citation

If you use `jax-rag-bench` in academic work, please cite:

```bibtex
@software{kohli_jaxragbench_2026,
  author  = {Kohli, Ved},
  title   = {jax-rag-bench: A reproducible benchmark harness for retrieval-augmented generation},
  year    = {2026},
  url     = {https://github.com/vkohli123/jax-rag-bench},
  version = {0.x}
}
```

<br />

## 🙏 Acknowledgements

This project stands on the shoulders of giants:

- [**ScaNN**](https://github.com/google-research/google-research/tree/master/scann) by Google Research — the ANN library that makes million-scale retrieval feel free
- [**Sentence-Transformers**](https://www.sbert.net/) by UKP-TUDA — the canonical encoder library
- [**BEIR**](https://github.com/beir-cellar/beir) — for the datasets and evaluation conventions
- [**JAX**](https://github.com/google/jax) — for making "just write NumPy and get a TPU" actually true
- [**pytrec_eval**](https://github.com/cvangysel/pytrec_eval) — our ground truth for IR metrics

Special thanks to early users who filed sharp bug reports — your `--verbose` logs made this software better.

<br />

## 📝 License

`jax-rag-bench` is released under the [Apache 2.0 License](./LICENSE). Use it, ship it, fork it, sell things built on it. A note in your acknowledgements is appreciated but not required.

<br />

---

<div align="center">

<sub>
Built with ❤️ in Bengaluru. Issues, ideas, and PRs welcome.<br/>
If this project saved you a weekend, <a href="https://github.com/vkohli123/jax-rag-bench">drop a ⭐</a> — it genuinely helps.
</sub>

</div>
