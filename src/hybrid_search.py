"""module for hybrid search combining dense and sparse embeddings."""

from src.generate_embeddings import make_sparse_embedding, make_dense_embedding
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SparseVector

from typing import Any
collection_name = "pubmedqa"

def hybrid_search(client, query_text: str, top_k: int = 10):
    """
    Perform hybrid search: run dense AND sparse search in parallel,
    return both result lists.
    """
    query_sparse_vectors = make_sparse_embedding([query_text])
    query_dense_vector = make_dense_embedding([query_text])

    search_results = client.query_batch_points(
        collection_name=collection_name,
        requests=[
            QueryRequest(
                query=query_dense_vector[0].tolist(),
                using="text-dense",
                limit=top_k,
                with_payload=True,
            ),
            QueryRequest(
                query=SparseVector(
                    indices=query_sparse_vectors[0].indices.tolist(),
                    values=query_sparse_vectors[0].values.tolist(),
                ),
                using="text-sparse",
                limit=top_k,
                with_payload=True,
            ),
        ],
    )

    return [search_results[0].points, search_results[1].points]

if __name__ == "__main__":
    from src.indexing import create_collection, make_points
    from src.generate_embeddings import generate_embeddings
    from src.data_cleaning_and_construction import build_document_text, construct_clean_dataset
    from src.load_dataset import load_and_prepare_dataset

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

    dense_results, sparse_results = search_results[0], search_results[1]

    print(f"Query: '{query_text}'")
    print(f"\n{'='*70}")
    print(f"DENSE (Semantic) Results — Top 5:")
    print(f"{'='*70}")
    for i, point in enumerate(dense_results[:5]):
        title = point.payload['text'][:100].replace('\n', ' ')
        print(f"  {i+1}. [Score: {point.score:.4f}] {title}...")

    print(f"\n{'='*70}")
    print(f"SPARSE (Lexical) Results — Top 5:")
    print(f"{'='*70}")
    for i, point in enumerate(sparse_results[:5]):
        title = point.payload['text'][:100].replace('\n', ' ')
        print(f"  {i+1}. [Score: {point.score:.4f}] {title}...")
