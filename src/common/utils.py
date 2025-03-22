import logging
import random
from typing import List

import nltk
import numpy as np
import torch
from nltk.corpus import brown

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sentences_from_brown_corpus() -> List[List[str]]:
    # download the brown corpus
    nltk.download('brown', quiet=True)
    logging.info("Loading and preprocessing Brown corpus...")
    sentences = brown.sents()
    sentences = [[word.lower() for word in sent] for sent in sentences]

    return sentences
