# ⚡ jax-rag-bench

<p align="center">
  <b>A clean, modular RAG benchmarking framework for evaluating retrieval quality with embeddings, ANN search, and ranking metrics.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/RAG-Benchmarking-purple" />
  <img src="https://img.shields.io/badge/Vector%20Search-ScaNN-orange" />
  <img src="https://img.shields.io/badge/Metrics-nDCG%40k-green" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" />
</p>

---

## 🚀 Overview

`jax-rag-bench` is a lightweight and extensible benchmarking project for evaluating the retrieval layer of Retrieval-Augmented Generation systems.

Modern RAG applications depend heavily on retrieval quality. Even if the language model is powerful, the final answer can still fail if the retriever returns weak, irrelevant, or poorly ranked context. This project focuses on that exact problem: **how do we measure and improve retrieval quality before generation happens?**

The project provides a modular pipeline for:

- generating document embeddings,
- building an Approximate Nearest Neighbor index,
- retrieving top-k relevant documents,
- evaluating ranking quality,
- and extending the system toward hybrid sparse-dense retrieval.

The goal is not only to build a working retrieval pipeline, but also to create a clean benchmarking foundation that can be expanded into larger RAG evaluation experiments.

---

## 🎯 Project Goal

The main goal of `jax-rag-bench` is to answer one practical question:

> **How good is our retriever at finding the right documents for a query?**

In real-world RAG systems, retrieval quality directly affects:

- answer correctness,
- hallucination reduction,
- context relevance,
- latency,
- user trust,
- and overall system reliability.

This project helps benchmark retrieval performance using measurable ranking metrics such as `nDCG@k`.

---

## 🧠 Why Retrieval Benchmarking Matters

A RAG pipeline usually has two major stages:

1. **Retriever** — finds relevant documents.
2. **Generator** — produces an answer using those documents.

Most people focus only on the generator. But in production systems, the retriever is often the real bottleneck.

If retrieval fails:

- the model receives irrelevant context,
- the answer quality drops,
- hallucinations increase,
- latency may rise due to unnecessary context,
- and evaluation becomes harder to debug.

That is why this project isolates the retrieval layer and evaluates it independently.

---

## 🏗️ Architecture

```txt
                 ┌────────────────────┐
                 │    Text Corpus      │
                 │   data/data.txt     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Embedding Pipeline  │
                 │ SentenceTransformers│
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  Vector Embeddings  │
                 │ dense representations│
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   ScaNN ANN Index   │
                 │ approximate search  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   Top-K Retrieval   │
                 │ nearest documents   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Evaluation Metrics  │
                 │ nDCG@k / Recall@k   │
                 └────────────────────┘
