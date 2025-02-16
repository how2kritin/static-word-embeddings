import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import brown
from typing import List, Dict, Tuple
import random
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Word2VecVocab:
    def __init__(self, sentences: List[List[str]], min_count: int = 5, max_vocab_size: int = 50000):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

        # Build vocabulary
        counter = Counter([word.lower() for sentence in sentences for word in sentence])

        # Filter by minimum count and vocabulary size
        filtered_words = {
            word: count for word, count in counter.most_common(max_vocab_size)
            if count >= min_count
        }

        # Create word to index mappings
        self.word2idx = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Create sampling table for negative sampling
        self.word_counts = np.array([count for count in filtered_words.values()])
        self.word_freqs = self.word_counts / np.sum(self.word_counts)
        self.word_freqs = np.power(self.word_freqs, 0.75)
        self.word_freqs = self.word_freqs / np.sum(self.word_freqs)

    def __len__(self):
        return len(self.word2idx)

    def get_index(self, word: str) -> int:
        return self.word2idx.get(word.lower(), -1)

    def get_word(self, idx: int) -> str:
        return self.idx2word.get(idx, "<UNK>")

    def sample_negative_words(self, batch_size: int, num_negative: int) -> torch.Tensor:
        """Sample negative words according to noise distribution"""
        negative_samples = np.random.choice(
            len(self),
            size=(batch_size, num_negative),
            p=self.word_freqs
        )
        return torch.LongTensor(negative_samples)


class CBOWDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Word2VecVocab, window_size: int = 5):
        self.vocab = vocab
        self.window_size = window_size
        self.data = []

        # Create training pairs with progress bar
        for sentence in tqdm(sentences, desc="Creating training pairs"):
            word_indices = [self.vocab.get_index(word) for word in sentence]
            word_indices = [idx for idx in word_indices if idx != -1]  # Remove unknown words

            for target_pos in range(len(word_indices)):
                context_indices = []
                for pos in range(max(0, target_pos - window_size),
                                 min(len(word_indices), target_pos + window_size + 1)):
                    if pos != target_pos:
                        context_indices.append(word_indices[pos])

                if context_indices:
                    while len(context_indices) < 2 * window_size:
                        context_indices.append(0)  # Pad with 0
                    self.data.append((context_indices, word_indices[target_pos]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.LongTensor(context), torch.LongTensor([target])


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOW, self).__init__()
        self.embeddings_input = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_output = nn.Embedding(vocab_size, embedding_dim)

        # Fix initialization
        initrange = 0.5 / embedding_dim
        self.embeddings_input.weight.data.uniform_(-initrange, initrange)
        self.embeddings_output.weight.data.uniform_(-initrange, initrange)

    def forward(self, contexts, targets, negative_samples):
        # Get context embeddings and average them
        context_embeds = self.embeddings_input(contexts).mean(dim=1)  # [batch_size, embed_dim]

        # Get positive target embeddings
        target_embeds = self.embeddings_output(targets).squeeze(1)  # [batch_size, embed_dim]

        # Get negative sample embeddings
        neg_embeds = self.embeddings_output(negative_samples)  # [batch_size, num_neg, embed_dim]

        # Compute scores using dot product
        pos_score = torch.sum(context_embeds * target_embeds, dim=1)  # [batch_size]
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, num_neg]

        return pos_score, neg_score


def train_word2vec(
        sentences: List[List[str]],
        embedding_dim: int = 100,
        window_size: int = 5,
        num_negative: int = 5,
        min_count: int = 5,
        batch_size: int = 32,
        num_epochs: int = 5,
        learning_rate: float = 0.001,
        save_path: str = 'word2vec_model.pt'
) -> Tuple[CBOW, Word2VecVocab]:
    # Create vocabulary
    logging.info("Building vocabulary...")
    vocab = Word2VecVocab(sentences, min_count=min_count)
    logging.info(f"Vocabulary size: {len(vocab)}")

    # Create dataset
    logging.info("Creating dataset...")
    dataset = CBOWDataset(sentences, vocab, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBOW(len(vocab), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with progress bars
    logging.info("Starting training...")
    epoch_bar = tqdm(range(num_epochs), desc="Training epochs")
    for epoch in epoch_bar:
        total_loss = 0
        batch_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                         desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch_idx, (contexts, targets) in batch_bar:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Generate negative samples
            negative_samples = vocab.sample_negative_words(
                contexts.size(0), num_negative
            ).to(device)

            # Forward pass
            pos_score, neg_score = model(contexts, targets, negative_samples)

            # Calculate loss with better numerical stability
            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pos_score,
                torch.ones_like(pos_score)
            )
            neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                neg_score,
                torch.zeros_like(neg_score)
            )
            loss = pos_loss + neg_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        epoch_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        logging.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    # Save model, embeddings and vocabulary
    save_model(model, vocab, save_path)

    return model, vocab


def save_model(model: CBOW, vocab: Word2VecVocab, save_path: str):
    """Save model, embeddings and vocabulary to file"""
    save_data = {
        'model_state_dict': model.state_dict(),
        'embeddings': model.embeddings_input.weight.detach().cpu(),
        'vocab_state': {
            'word2idx': vocab.word2idx,
            'idx2word': vocab.idx2word,
            'word_freqs': vocab.word_freqs,
            'word_counts': vocab.word_counts
        },
        'metadata': {
            'embedding_dim': model.embeddings_input.weight.shape[1],
            'vocab_size': len(vocab),
            'model_type': 'cbow'
        }
    }
    torch.save(save_data, save_path)
    logging.info(f"Saved model to {save_path}")


def load_model(load_path: str, device=None) -> Tuple[CBOW, Word2VecVocab, Dict]:
    """Load saved model, embeddings and vocabulary"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_data = torch.load(load_path, map_location=device)

    # Reconstruct vocabulary
    vocab = Word2VecVocab([], min_count=1)  # Create empty vocab
    vocab.word2idx = save_data['vocab_state']['word2idx']
    vocab.idx2word = save_data['vocab_state']['idx2word']
    vocab.word_freqs = save_data['vocab_state']['word_freqs']
    vocab.word_counts = save_data['vocab_state']['word_counts']

    # Reconstruct model
    model = CBOW(len(vocab), save_data['metadata']['embedding_dim']).to(device)
    model.load_state_dict(save_data['model_state_dict'])

    return model, vocab, save_data['metadata']


def find_similar_words(word: str, vocab: Word2VecVocab, model: CBOW, top_k: int = 5) -> List[Tuple[str, float]]:
    """Find most similar words using cosine similarity"""
    if word not in vocab.word2idx:
        return []

    device = next(model.parameters()).device
    word_idx = vocab.word2idx[word]
    embeddings = model.embeddings_input.weight.detach()
    word_vec = embeddings[word_idx]

    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        word_vec.unsqueeze(0),
        embeddings,
        dim=1
    )

    # Get top k similar words (excluding the input word)
    top_indices = similarities.argsort(descending=True)

    most_similar = []
    for idx in top_indices[1:top_k + 1]:  # Skip the first one as it's the input word
        most_similar.append((vocab.get_word(idx.item()), similarities[idx].item()))

    return most_similar


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Download Brown corpus if needed
    nltk.download('brown', quiet=True)

    # Load and preprocess Brown corpus
    logging.info("Loading and preprocessing Brown corpus...")
    sentences = brown.sents()
    sentences = [[word.lower() for word in sent] for sent in sentences]

    # Train word2vec model
    model, vocab = train_word2vec(
        sentences,
        embedding_dim=300,
        window_size=3,
        num_negative=5,
        min_count=3,
        batch_size=256,
        num_epochs=5,
        learning_rate=0.001,
        save_path='brown_word2vec.pt'
    )

    # Load trained model
    model, vocab, metadata = load_model('brown_word2vec.pt')

    # Test similar words with progress bar
    test_words = ['king', 'queen', 'man', 'woman', 'city', 'computer']
    for word in tqdm(test_words, desc="Testing similar words"):
        similar_words = find_similar_words(word, vocab, model)
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")


if __name__ == "__main__":
    main()