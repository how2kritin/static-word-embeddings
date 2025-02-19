import logging
from typing import Tuple, List, Literal

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.save_and_load import save_model
from src.data_processing.Word2VecVocab import Word2VecVocab
from src.data_processing.datasets import SkipgramDataset, CBOWDataset
from src.models.cbow import CBOW
from src.models.skipgram import SkipGram


def train_model(model, vocab, dataloader, num_epochs, num_negative, device, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logging.info("Starting training...")
    epoch_bar = tqdm(range(num_epochs), desc="Training epochs")
    for epoch in epoch_bar:
        total_loss = 0
        batch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                         leave=False)

        for batch_idx, (contexts, targets) in batch_bar:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # generate negative samples
            negative_samples = vocab.sample_negative_words(contexts.size(0), num_negative).to(device)
            pos_score, neg_score = model(contexts, targets, negative_samples)

            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        epoch_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        logging.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    return model


def train_handler(model_type: Literal['cbow', 'skipgram'], sentences: List[List[str]], embedding_dim: int = 100,
                  window_size: int = 5, num_negative: int = 5, min_count: int = 5, batch_size: int = 32,
                  num_epochs: int = 5, learning_rate: float = 0.001, save_path: str = 'model.pt') -> Tuple[
    CBOW | SkipGram, Word2VecVocab]:
    logging.info("Building vocabulary...")
    vocab = Word2VecVocab(sentences, min_count=min_count)
    logging.info(f"Vocabulary size: {len(vocab)}")

    logging.info("Creating dataset...")
    dataset = {'cbow': CBOWDataset(sentences, vocab, window_size=window_size),
               'skipgram': SkipgramDataset(sentences, vocab, window_size=window_size)}[model_type]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {'cbow': CBOW(len(vocab), embedding_dim).to(device),
             'skipgram': SkipGram(len(vocab), embedding_dim).to(device)}[model_type]

    train_model(model, vocab, dataloader, num_epochs, num_negative, device, learning_rate)

    save_model(model, vocab, save_path)

    return model, vocab
