from collections import Counter
from typing import List

import numpy as np
import torch


class Word2VecVocab:
    def __init__(self, sentences: List[List[str]], min_count: int = 5, max_vocab_size: int = 50000):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

        counter = Counter([word.lower() for sentence in sentences for word in sentence])

        filtered_words = {word: count for word, count in counter.most_common(max_vocab_size) if count >= min_count}

        self.word2idx = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.word_counts = np.array([count for count in filtered_words.values()])
        self.word_freqs = self.word_counts / np.sum(self.word_counts)
        self.word_freqs = np.power(self.word_freqs, 0.75)
        self.word_freqs = self.word_freqs / np.sum(self.word_freqs)

    def __len__(self):
        return len(self.word2idx)

    def sample_negative_words(self, batch_size: int, num_negative: int) -> torch.Tensor:
        negative_samples = np.random.choice(len(self), size=(batch_size, num_negative), p=self.word_freqs)
        return torch.LongTensor(negative_samples)
