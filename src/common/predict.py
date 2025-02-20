from typing import Tuple, List, Any

import torch


def find_similar_words(word: str, word2idx: dict, embeddings: Any, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    To find the most similar words using cosine similarity.
    """
    if word not in word2idx:
        return []

    idx2word = {idx: word for word, idx in word2idx.items()}
    word_vec = embeddings[word2idx]

    similarities = torch.nn.functional.cosine_similarity(word_vec.unsqueeze(0), embeddings, dim=1)
    top_indices = similarities.argsort(descending=True)

    most_similar = []
    for idx in top_indices[1:top_k + 1]:  # skip the first one as that's just the input word
        most_similar.append((idx2word[idx.item()], similarities[idx].item()))

    return most_similar
