import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


def load_wordsim353crowd() -> DataFrame:
    """
    Load the WordSim353Crowd dataset.
    Returns a pandas DataFrame with columns: Word 1, Word 2, Human (Mean)
    """
    dataset_path = "corpus/wordsim353crowd.csv"
    df = pd.read_csv(dataset_path, delimiter=',')
    return df


def load_embeddings(path) -> dict:
    """
    Load PyTorch word embeddings with vocabulary mapping
    """
    saved_data = torch.load(path)
    embeddings = saved_data['embeddings']
    vocab_state = saved_data['vocab_state']
    word2idx = vocab_state['word2idx']

    embeddings_dict = {}
    for word, idx in word2idx.items():
        embeddings_dict[word] = embeddings[idx]

    print(f"Loaded embeddings with dimensions: {embeddings.shape}")
    print(f"Vocabulary size: {len(word2idx)}")

    return embeddings_dict


def compute_cosine_similarity(vec1, vec2) -> float:
    """
    Compute the cosine similarity between two vectors
    """
    if torch.is_tensor(vec1):
        vec1 = vec1.cpu().numpy()
    if torch.is_tensor(vec2):
        vec2 = vec2.cpu().numpy()
    return 1 - cosine(vec1, vec2)


def compute_similarities(word_pairs, embeddings_dict):
    """
    Compute cosine similarities between pairs of words using their embeddings.
    """
    similarities = []
    valid_indices = []
    skipped_pairs = []

    for idx, row in word_pairs.iterrows():
        word1, word2 = row['Word 1'].lower(), row['Word 2'].lower()

        if word1 in embeddings_dict and word2 in embeddings_dict:
            vec1 = embeddings_dict[word1]
            vec2 = embeddings_dict[word2]

            similarity = compute_cosine_similarity(vec1, vec2)
            similarities.append(similarity)
            valid_indices.append(idx)
        else:
            missing_words = []
            if word1 not in embeddings_dict:
                missing_words.append(word1)
            if word2 not in embeddings_dict:
                missing_words.append(word2)
            skipped_pairs.append((word1, word2, missing_words))

    return np.array(similarities), valid_indices, skipped_pairs


def main(inp_path: str):
    print("Loading WordSim353Crowd dataset...")
    try:
        wordsim_df = load_wordsim353crowd()
        print(f"Loaded {len(wordsim_df)} word pairs from WordSim353Crowd")
    except Exception as e:
        print(f"Error loading WordSim353Crowd dataset: {e}")
        return

    print("\nLoading word embeddings...")
    try:
        embeddings_dict = load_embeddings(inp_path)
    except Exception as e:
        print(f"Error loading word embeddings: {e}")
        return

    print("\nComputing cosine similarities...")
    computed_similarities, valid_indices, skipped_pairs = compute_similarities(wordsim_df, embeddings_dict)

    if len(valid_indices) == 0:
        print("No valid word pairs found in the embeddings!")
        return

    human_scores = wordsim_df.iloc[valid_indices]['Human (Mean)'].values

    # calculate Spearman's correlation between computed Cosine Similarities and human scores.
    correlation, _ = spearmanr(computed_similarities, human_scores)


    print("\nResults:")
    print(f"Number of valid word pairs: {len(valid_indices)}")
    print(f"Number of skipped pairs: {len(skipped_pairs)}")
    print(f"Spearman's correlation: {correlation:.4f}")
    results_df = pd.DataFrame(
        {'Word 1': wordsim_df.iloc[valid_indices]['Word 1'], 'Word 2': wordsim_df.iloc[valid_indices]['Word 2'],
            'Human (Mean)': human_scores, 'Cosine Similarity': computed_similarities})

    results_file = 'similarity_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults have been saved to '{results_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("we", type=str, help="Path to the word embeddings.")
    args = parser.parse_args()
    main(args.we)
