import logging
from typing import List

import nltk
import torch
from nltk.corpus import brown


def get_sentences_from_brown_corpus() -> List[List[str]]:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # download the brown corpus
    nltk.download('brown', quiet=True)
    logging.info("Loading and preprocessing Brown corpus...")
    sentences = brown.sents()
    sentences = [[word.lower() for word in sent] for sent in sentences]

    return sentences
