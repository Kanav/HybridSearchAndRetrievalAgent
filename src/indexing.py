"""module for indexing embeddings."""

import pandas as pd
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

collection_name = "pubmedqa"

def create_collection():
    print("This function would create a collection in the vector database.")
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name,
        vectors_config={
            "text-dense": VectorParams(
                size=1024,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )

    print(f"✅ Collection '{collection_name}' created with dense + sparse vector support")
    return client

def make_points(df: pd.DataFrame) -> list[PointStruct]:
    """Convert DataFrame rows into Qdrant PointStruct objects."""
    sparse_vectors = df["sparse_embedding"].tolist()
    doc_texts = df["combined_text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    rows = df.to_dict(orient="records")

    points = []
    for idx, (text, sparse_vector, dense_vector) in enumerate(
            zip(doc_texts, sparse_vectors, dense_vectors)
    ):
        sparse_vector = SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist()
        )

        point = PointStruct(
            id=idx,
            payload={
                "text": text,
                "pubid": int(rows[idx]["pubid"]),
                "question": rows[idx]["question"],
                "final_decision": rows[idx]["final_decision"],
            },
            vector={
                "text-sparse": sparse_vector,
                "text-dense": dense_vector.tolist(),
            },
        )
        points.append(point)
    return points

if __name__ == "__main__":
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
    client.upsert(collection_name, points)
    print(f"✅ {len(points)} points indexed in Qdrant!")
