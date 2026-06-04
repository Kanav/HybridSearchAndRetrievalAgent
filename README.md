## 🧠 Hybrid Search and Retrieval Agent

### 📘 Project Overview

The **Hybrid Search and Retrieval Agent** is a modular system that combines **dense (semantic)** and **sparse (lexical)** retrieval techniques to perform high-quality information retrieval. It leverages **FastEmbed** for embedding generation and **Qdrant** as the vector database backend.

The project demonstrates how to build a **hybrid search pipeline** that integrates both **semantic understanding** (via dense embeddings) and **keyword matching** (via sparse embeddings) to improve retrieval accuracy across diverse query types.

---

### 🏗️ High-Level Architecture

Below is a simplified ASCII-style architecture diagram for better visibility across all markdown renderers:

```
┌────────────────────────────┐
│        Load Dataset        │
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│ Clean & Construct Text Data│
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│     Generate Embeddings     │
│ ┌─────────────────────────┐ │
│ │ Sparse (SPLADE)        │ │
│ │ Dense (BGE-Large)      │ │
│ └─────────────────────────┘ │
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│       Index in Qdrant      │
│ ┌─────────────────────────┐ │
│ │ Dense Vector Index      │ │
│ │ Sparse Vector Index     │ │
│ └─────────────────────────┘ │
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│       Hybrid Search        │
└────────────┬───────────────┘
             │
┌────────────▼───────────────┐
│ Results Fusion (Optional)  │
│     via RRF Algorithm      │
└────────────────────────────┘
```

---

### ⚙️ Setup and Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/HybridSearchAndRetrievalAgent.git
cd HybridSearchAndRetrievalAgent
```

#### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
# or if using pyproject.toml
pip install .
```

#### 4. (Optional) Install Qdrant locally or use Qdrant Cloud
- Local: [https://qdrant.tech/documentation/quick_start/](https://qdrant.tech/documentation/quick_start/)
- Cloud: [https://cloud.qdrant.io/](https://cloud.qdrant.io/)

---

### ▶️ Running the Project

#### Step 1: Load and preprocess dataset
```bash
python -m src.load_dataset
```

#### Step 2: Generate embeddings
```bash
python -m src.generate_embeddings
```

#### Step 3: Create Qdrant collection and index data
```bash
python -m src.indexing
```

#### Step 4: Run hybrid search
```bash
python -m src.hybrid_search
```

---

### 🧩 Main Modules and Their Functions

| Module | Description |
|--------|--------------|
| [`src/load_dataset.py`](src/load_dataset.py) | Loads datasets (e.g., AG News, PubMedQA) using Hugging Face Datasets API. |
| [`src/data_cleaning_and_construction.py`](src/data_cleaning_and_construction.py) | Cleans and constructs text fields for embedding. |
| [`src/generate_embeddings.py`](src/generate_embeddings.py) | Generates **sparse (SPLADE)** and **dense (BGE-Large)** embeddings using FastEmbed. |
| [`src/indexing.py`](src/indexing.py) | Creates a **Qdrant collection** and indexes documents with both dense and sparse vectors. |
| [`src/hybrid_search.py`](src/hybrid_search.py) | Performs **hybrid search** combining dense and sparse retrieval results. |
| [`src/rrf.py`](src/rrf.py) | Implements **Reciprocal Rank Fusion (RRF)** for combining search results. |
| [`src/main.py`](src/main.py) | Orchestrates the full pipeline from data loading to hybrid search execution. |

---

### 🔍 Example Queries

Example hybrid search query:
```python
from qdrant_client import QdrantClient
from src.hybrid_search import hybrid_search

client = QdrantClient(':memory:')
query = "Does metformin improve insulin sensitivity in type 2 diabetes?"

dense_results, sparse_results = hybrid_search(client, query)

print("Top Dense Results:")
for i, point in enumerate(dense_results[:5]):
    print(f"{i+1}. {point.payload['text'][:100]} (Score: {point.score:.4f})")

print("\nTop Sparse Results:")
for i, point in enumerate(sparse_results[:5]):
    print(f"{i+1}. {point.payload['text'][:100]} (Score: {point.score:.4f})")
```

---

### 🧱 Dependencies

- **Python 3.9+**
- **FastEmbed** — for generating dense and sparse embeddings
- **Qdrant Client** — for vector database operations
- **Pandas** — for data manipulation
- **Hugging Face Datasets** — for dataset loading

Install all dependencies via:
```bash
pip install fastembed qdrant-client pandas datasets
```

---

### ⚙️ Configuration and Environment

- Default collection name: `pubmedqa`
- Dense model: `BAAI/bge-large-en-v1.5`
- Sparse model: `prithvida/Splade_PP_en_v1`
- Embeddings are generated in batches of 32 for efficiency.

---

### 🧩 Example Workflow Summary

1. Load dataset → Clean text → Build combined text field.
2. Generate dense and sparse embeddings.
3. Create Qdrant collection with dual vector configuration.
4. Index all documents.
5. Perform hybrid search combining both retrieval modes.
6. (Optional) Fuse results using RRF for improved ranking.

---

### 📄 License

This project is open-source and available under the **MIT License**.

---
