from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data_processing.Word2VecVocab import Word2VecVocab


class CBOWDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Word2VecVocab, window_size: int = 5):
        self.vocab = vocab
        self.window_size = window_size
        self.data = []

        for sentence in tqdm(sentences, desc="Creating training pairs"):
            word_indices = [self.vocab.word2idx.get(word.lower(), -1) for word in sentence]
            word_indices = [idx for idx in word_indices if idx != -1]  # remove unknown words

            for target_pos in range(len(word_indices)):
                context_indices = []
                for pos in range(max(0, target_pos - window_size),
                                 min(len(word_indices), target_pos + window_size + 1)):
                    if pos != target_pos:
                        context_indices.append(word_indices[pos])

                if context_indices:
                    while len(context_indices) < 2 * window_size:
                        context_indices.append(0)  # pad with 0 (essentially, that idx is treated to be <PAD>)
                    self.data.append((context_indices, word_indices[target_pos]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.LongTensor(context), torch.LongTensor([target])


class SkipgramDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Word2VecVocab, window_size: int = 5):
        self.vocab = vocab
        self.window_size = window_size
        self.data = []

        for sentence in tqdm(sentences, desc="Creating training pairs"):
            word_indices = [self.vocab.word2idx.get(word.lower(), -1) for word in sentence]
            word_indices = [idx for idx in word_indices if idx != -1]  # remove unknown words

            for center_pos in range(len(word_indices)):
                # for each context position
                for context_pos in range(max(0, center_pos - window_size),
                                         min(len(word_indices), center_pos + window_size + 1)):
                    if context_pos != center_pos:
                        self.data.append((word_indices[center_pos],  # center word
                                          word_indices[context_pos]  # context word
                                          ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.LongTensor([center]), torch.LongTensor([context])
