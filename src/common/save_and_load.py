import logging
from typing import Tuple, Dict, Literal

import torch

from src.data_processing.Word2VecVocab import Word2VecVocab
from src.models.cbow import CBOW
from src.models.skipgram import SkipGram


def save_model(model: CBOW | SkipGram, vocab: Word2VecVocab, save_path: str) -> None:
    save_data = {'model_state_dict': model.state_dict(), 'embeddings': model.embeddings_input.weight.detach().cpu(),
                 'vocab_state': {'word2idx': vocab.word2idx, 'idx2word': vocab.idx2word, 'word_freqs': vocab.word_freqs,
                                 'word_counts': vocab.word_counts},
                 'metadata': {'embedding_dim': model.embeddings_input.weight.shape[1], 'vocab_size': len(vocab),
                              'model_type': type(model).__name__}}
    torch.save(save_data, save_path)
    logging.info(f"Saved model to {save_path}")


def load_model(model_type: Literal['cbow', 'skipgram'], load_path: str, device=None) -> Tuple[
    CBOW | SkipGram, Word2VecVocab, Dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_data = torch.load(load_path, map_location=device)

    vocab = Word2VecVocab([], min_count=1)
    vocab.word2idx = save_data['vocab_state']['word2idx']
    vocab.idx2word = save_data['vocab_state']['idx2word']
    vocab.word_freqs = save_data['vocab_state']['word_freqs']
    vocab.word_counts = save_data['vocab_state']['word_counts']

    model = {'cbow': CBOW(len(vocab), save_data['metadata']['embedding_dim']).to(device),
             'skipgram': SkipGram(len(vocab), save_data['metadata']['embedding_dim']).to(device)}[model_type]
    model.load_state_dict(save_data['model_state_dict'])

    return model, vocab, save_data['metadata']
