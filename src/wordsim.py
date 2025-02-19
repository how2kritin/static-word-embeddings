import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')


def load_wordsim353():
    """
    Load the WordSim353 dataset.
    Returns a pandas DataFrame with columns: word1, word2, similarity_score
    """
    dataset_path = "corpus/wordsim353crowd.csv"
    df = pd.read_csv(dataset_path, delimiter=',')
    return df


def load_embeddings(path):
    """
    Load PyTorch word embeddings with vocabulary mapping
    """
    saved_data = torch.load(path)
    embeddings = saved_data['embeddings']  # This is the embedding matrix
    vocab_state = saved_data['vocab_state']
    word2idx = vocab_state['word2idx']

    # Create a dictionary mapping words to their embeddings
    embeddings_dict = {}
    for word, idx in word2idx.items():
        embeddings_dict[word] = embeddings[idx]

    print(f"Loaded embeddings with dimensions: {embeddings.shape}")
    print(f"Vocabulary size: {len(word2idx)}")

    return embeddings_dict


def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors
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

        # Check if both words exist in the vocabulary
        if word1 in embeddings_dict and word2 in embeddings_dict:
            # Get word vectors
            vec1 = embeddings_dict[word1]
            vec2 = embeddings_dict[word2]

            # Compute cosine similarity
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


def main():
    # Load WordSim353 dataset
    print("Loading WordSim353 dataset...")
    try:
        wordsim_df = load_wordsim353()
        print(f"Loaded {len(wordsim_df)} word pairs from WordSim353")
    except Exception as e:
        print(f"Error loading WordSim353 dataset: {e}")
        return

    # Load word embeddings
    print("\nLoading word embeddings...")
    inp_path = input("Please provide path to word embeddings: ")
    try:
        embeddings_dict = load_embeddings(inp_path)
    except Exception as e:
        print(f"Error loading word embeddings: {e}")
        return

    # Compute cosine similarities
    print("\nComputing cosine similarities...")
    computed_similarities, valid_indices, skipped_pairs = compute_similarities(wordsim_df, embeddings_dict)

    if len(valid_indices) == 0:
        print("No valid word pairs found in the embeddings!")
        return

    # Get the human-annotated scores for valid pairs
    human_scores = wordsim_df.iloc[valid_indices]['Human (Mean)'].values

    # Calculate Spearman's correlation
    correlation, _ = spearmanr(computed_similarities, human_scores)

    # Print results
    print("\nResults:")
    print(f"Number of valid word pairs: {len(valid_indices)}")
    print(f"Number of skipped pairs: {len(skipped_pairs)}")
    print(f"Spearman's correlation: {correlation:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Word1': wordsim_df.iloc[valid_indices]['Word 1'],
        'Word2': wordsim_df.iloc[valid_indices]['Word 2'],
        'Human_Score': human_scores,
        'Computed_Similarity': computed_similarities
    })

    # Save results
    results_file = 'similarity_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults have been saved to '{results_file}'")

    # Save skipped pairs with more detailed information
    if skipped_pairs:
        skipped_df = pd.DataFrame(skipped_pairs, columns=['Word 1', 'Word 2', 'Missing_Words'])
        skipped_file = 'skipped_pairs.csv'
        skipped_df.to_csv(skipped_file, index=False)
        print(f"Skipped pairs have been saved to '{skipped_file}'")

        # Print some statistics about skipped pairs
        print("\nSkipped pairs statistics:")
        total_missing_words = sum(len(missing) for _, _, missing in skipped_pairs)
        print(f"Total missing words: {total_missing_words}")
        print(f"Average missing words per skipped pair: {total_missing_words / len(skipped_pairs):.2f}")


if __name__ == "__main__":
    main()