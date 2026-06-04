"""module for generating embeddings from text data."""

import pandas as pd
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

print(f"FastEmbed version: {fastembed.__version__}")
sparse_model_name = "prithvida/Splade_PP_en_v1"
dense_model_name = "BAAI/bge-large-en-v1.5"

sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)
print("✅ Both models loaded!")

def generate_embeddings(df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:
    import time

    product_texts = df[text_column].tolist()
    print(f"Embedding {len(product_texts)} documents...")

    start = time.time()
    print("⏳ Generating sparse embeddings (SPLADE)...")
    df["sparse_embedding"] = make_sparse_embedding(product_texts)
    print(f"✅ Done in {time.time() - start:.2f}s!")

    start = time.time()
    print("⏳ Generating dense embeddings (BGE-Large)...")
    df["dense_embedding"] = make_dense_embedding(product_texts)
    print(f"✅ Done in {time.time() - start:.2f}s!")

    return df

# Helper functions
def make_sparse_embedding(texts: list[str]) -> list[SparseEmbedding]:
    return list(sparse_model.embed(texts, batch_size=32))

def make_dense_embedding(texts: list[str]):
    return list(dense_model.embed(texts))


if __name__ == "__main__":
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

    print("✅ Embedding generation complete.")
