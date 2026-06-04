"""Module for cleaning and constructing datasets for the Hybrid Search and Retrieval Agent."""

import pandas as pd
import numpy as np


def construct_clean_dataset(df: pd.DataFrame, text_column: str = "text", label_column: str = None) -> pd.DataFrame:
    # Check for duplicate pubids
    print(f"Total rows:           {len(df)}")
    print(f"Unique pubids:        {df['pubid'].nunique()}")
    print(f"Duplicate pubids:     {df.duplicated(subset=['pubid']).sum()}")

    print(f"\n--- Missing Values ---")
    print(df.isnull().sum())

    print(f"\n--- Empty Strings ---")
    print(f"Empty questions:      {(df['question'] == '').sum()}")
    print(f"Empty long_answers:   {(df['long_answer'] == '').sum()}")

    print(f"\n--- Final Decision Distribution ---")
    print(df['final_decision'].value_counts())

def build_document_text(df: pd.DataFrame) -> pd.DataFrame:
    def _combine_row(row) -> str:
        parts = []

        context = row["context"]
        if isinstance(context, str):
            context = json.loads(context)

        for para, label in zip(context["contexts"], context["labels"]):
            parts.append(f"[{label}] {para}")

        parts.append(f"[CONCLUSION] {row['long_answer']}")

        mesh_terms = context.get("meshes", [])
        if len(mesh_terms) > 0:
            parts.append(f"[MeSH Terms] {', '.join(mesh_terms)}")

        return "\n\n".join(parts)

    df["combined_text"] = df.apply(_combine_row, axis=1)

    print(f"Total documents to embed: {len(df)}")
    print(f"\nSample document (first 500 chars):")
    print(df["combined_text"].iloc[0][:500] + "...")

    return df

if __name__ == "__main__":
    from src.load_dataset import load_and_prepare_dataset

    print("Loading dataset using Hugging Face Datasets API...")
    df = load_and_prepare_dataset(dataset_name="ag_news", split="train")

    print(f"Loaded dataset with {len(df)} rows. Cleaning and analyzing data...")
    construct_clean_dataset(df, text_column="text", label_column="label")
    df = build_document_text(df)
