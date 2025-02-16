import logging
import os
from collections import Counter
from typing import List, Tuple

import nltk
import torch
from nltk.corpus import brown
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WordEmbeddingSVD:
    def __init__(self, window_size: int = 2, min_freq: int = 5, embedding_dim: int = 100):
        self.window_size = window_size
        self.min_freq = min_freq
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def preprocess_corpus(self, sentences: List[List[str]]) -> List[List[str]]:
        logging.info("Preprocessing corpus...")
        word_freq = Counter()
        for sent in tqdm(sentences, desc="Counting words"):
            word_freq.update([word.lower() for word in sent])

        vocab = [word for word, freq in word_freq.items() if freq >= self.min_freq]
        logging.info(f"Vocabulary size: {len(vocab)} words (min_freq={self.min_freq})")

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        processed = []
        for sent in tqdm(sentences, desc="Processing sentences"):
            processed.append([word.lower() for word in sent])
        return processed

    def build_cooccurrence_matrix(self, sentences: List[List[str]]) -> torch.Tensor:
        logging.info("Building co-occurrence matrix...")
        vocab_size = len(self.word2idx)
        cooccurrence = torch.zeros((vocab_size, vocab_size), device=self.device)

        for sentence in tqdm(sentences, desc="Building co-occurrence matrix"):
            indices = [self.word2idx[word] for word in sentence if word in self.word2idx]

            for center_idx, center_word_idx in enumerate(indices):
                context_idx_start = max(0, center_idx - self.window_size)
                context_idx_end = min(len(indices), center_idx + self.window_size + 1)

                for context_idx in range(context_idx_start, context_idx_end):
                    if center_idx != context_idx:
                        context_word_idx = indices[context_idx]
                        distance = abs(center_idx - context_idx)
                        cooccurrence[center_word_idx][context_word_idx] += 1.0 / distance

        logging.info(f"Co-occurrence matrix shape: {cooccurrence.shape}")
        return cooccurrence

    def apply_svd(self, cooccurrence_matrix: torch.Tensor) -> torch.Tensor:
        logging.info("Applying SVD...")

        # Process the log operation in chunks
        chunk_size = 1024
        rows, cols = cooccurrence_matrix.shape
        log_cooccurrence = torch.empty_like(cooccurrence_matrix)

        logging.info("Computing log transform...")
        for i in tqdm(range(0, rows, chunk_size)):
            i_end = min(i + chunk_size, rows)
            log_cooccurrence[i:i_end] = torch.log(cooccurrence_matrix[i:i_end] + 1e-8)
            torch.cuda.empty_cache()

        logging.info("Computing SVD...")
        try:
            U, S, V = torch.linalg.svd(log_cooccurrence, full_matrices=False)
            embeddings = U[:, :self.embedding_dim]
        except RuntimeError as e:
            logging.warning(f"SVD failed on GPU: {e}")
            logging.info("Falling back to CPU for SVD computation...")
            cpu_matrix = log_cooccurrence.cpu()
            U, S, V = torch.linalg.svd(cpu_matrix, full_matrices=False)
            embeddings = U[:, :self.embedding_dim].to(self.device)

        logging.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings

    def train(self, corpus: List[List[str]]) -> torch.Tensor:
        logging.info("Starting training process...")
        processed_corpus = self.preprocess_corpus(corpus)
        cooccurrence_matrix = self.build_cooccurrence_matrix(processed_corpus)
        self.embeddings = self.apply_svd(cooccurrence_matrix)
        logging.info("Training completed!")
        return self.embeddings

    def get_word_vector(self, word: str) -> torch.Tensor:
        if word not in self.word2idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.embeddings[self.word2idx[word]]

    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words using cosine similarity"""
        if word not in self.word2idx:
            return []

        word_idx = self.word2idx[word]
        word_vec = self.embeddings[word_idx]

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(word_vec.unsqueeze(0), self.embeddings, dim=1)

        # Get top k similar words (excluding the input word)
        top_indices = similarities.argsort(descending=True)

        most_similar = []
        for idx in top_indices[1:top_k + 1]:  # Skip the first one as it's the input word
            most_similar.append((self.idx2word[idx.item()], similarities[idx].item()))

        return most_similar

    def save(self, path: str):
        """Save SVD model, embeddings and vocabulary to file in the same format as Skip-gram"""
        save_data = {
            'embeddings': self.embeddings.detach().cpu(),
            'vocab_state': {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
            },
            'metadata': {
                'embedding_dim': self.embedding_dim,
                'vocab_size': len(self.word2idx),
                'model_type': 'svd'
            }
        }
        torch.save(save_data, path)
        logging.info(f"Saved SVD model to {path}")

    @classmethod
    def load(cls, path: str):
        """Load saved SVD model, embeddings and vocabulary"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        save_data = torch.load(path, map_location=device)

        # Create instance with saved metadata
        instance = cls(
            window_size=3,  # Default values since they're not needed for inference
            min_freq=3,
            embedding_dim=save_data['metadata']['embedding_dim']
        )

        # Load model state
        instance.embeddings = save_data['embeddings'].to(device)

        # Load vocabulary state
        vocab_state = save_data['vocab_state']
        instance.word2idx = vocab_state['word2idx']
        instance.idx2word = vocab_state['idx2word']

        # Log loading info
        logging.info(f"Model loaded from {path}")
        logging.info(f"Loaded model type: {save_data['metadata']['model_type']}")
        logging.info(f"Vocabulary size: {save_data['metadata']['vocab_size']}")
        logging.info(f"Embedding dimension: {save_data['metadata']['embedding_dim']}")

        return instance


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Download Brown corpus if needed
    nltk.download('brown', quiet=True)

    # Load and preprocess Brown corpus
    sentences = brown.sents()
    sentences = [[word.lower() for word in sent] for sent in sentences]
    logging.info(f"Loaded {len(sentences)} sentences from Brown corpus")

    # Initialize and train model
    model = WordEmbeddingSVD(window_size=3, min_freq=3, embedding_dim=300)
    embeddings = model.train(sentences)

    # Print final statistics
    logging.info("\nFinal Statistics:")
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"Embeddings device: {embeddings.device}")
    if torch.cuda.is_available():
        logging.info(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
    else:
        logging.info("Running on CPU")

    # Save the model
    model.save('svd.pt')

    # Test loading and similar words
    loaded_model = WordEmbeddingSVD.load('svd.pt')
    logging.info(f"Loaded embeddings shape: {loaded_model.embeddings.shape}")

    # Test similar words
    test_words = ['king', 'queen', 'man', 'woman', 'city']
    for word in tqdm(test_words, desc="Testing similar words"):
        similar_words = loaded_model.find_similar_words(word)
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")


if __name__ == "__main__":
    main()
