"""Module to load and preprocess datasets for the Hybrid Search and Retrieval Agent."""

import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer



def load_and_prepare_dataset(dataset_name: str = "ag_news", split: str = "train") -> pd.DataFrame:
    """Load a dataset from the Hugging Face Hub and return it as a pandas DataFrame.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The dataset split to load (e.g., 'train', 'test').

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    # Load the PubMedQA artificial split
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"Total instances: {len(dataset)}")
    print(f"Features: {dataset.column_names}")
    dataset = dataset.select(range(100))
    print(f"Working with {len(dataset)} instances")
    df = dataset.to_pandas()
    return df


if __name__ == "__main__":
    df = load_and_prepare_dataset()
    print(df.head())
