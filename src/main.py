import numpy as np
import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVector,
    PointStruct,
    QueryRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)
from transformers import AutoTokenizer

import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

from src.indexing import create_collection, make_points
from src.generate_embeddings import generate_embeddings
from src.data_cleaning_and_construction import build_document_text, construct_clean_dataset
from src.load_dataset import load_and_prepare_dataset
from src.hybrid_search import hybrid_search
from src.rrf import rank_list,rrf

def main() -> None:
    collection_name = "pubmedqa"
    print("Loading dataset using Hugging Face Datasets API...")
    df = load_and_prepare_dataset(dataset_name="ag_news", split="train")

    print(f"Loaded dataset with {len(df)} rows. Cleaning and analyzing data...")
    construct_clean_dataset(df, text_column="text", label_column="label")

    print("Building document text for embedding...")
    df = build_document_text(df)

    print("Generating embeddings...")
    df = generate_embeddings(df)

    print("Creating Qdrant collection and indexing points...")
    client = create_collection()
    points = make_points(df)
    print(f"Prepared {len(points)} points for indexing")
    client.upsert("pubmedqa", points)
    print(f"✅ {len(points)} points indexed in Qdrant!")

    print("Performing hybrid search (placeholder)...")

    # Query 1: Technical / specific — exact drug name
    query_text = "Does metformin improve insulin sensitivity in type 2 diabetes?"
    search_results = hybrid_search(client, query_text)

    dense_rank_list = rank_list(search_results[0])
    sparse_rank_list = rank_list(search_results[1])

    rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])

    print(f"\n🏆 HYBRID SEARCH RESULTS for: '{query_text}'")
    print(f"{'='*70}")

    fused_records = client.retrieve(
        collection_name=collection_name,
        ids=[item[0] for item in rrf_rank_list]
    )

    record_by_id = {r.id: r for r in fused_records}

    for rank, (item_id, score) in enumerate(rrf_rank_list, 1):
        record = record_by_id[item_id]
        title = record.payload['text'][:80].replace('\n', ' ')

        in_dense = any(id == item_id for id, _ in dense_rank_list)
        in_sparse = any(id == item_id for id, _ in sparse_rank_list)
        source = "BOTH" if (in_dense and in_sparse) else ("Dense only" if in_dense else "Sparse only")

        print(f"  {rank:>2}. [{source:<12}] {title}...")


if __name__ == "__main__":
    main()

