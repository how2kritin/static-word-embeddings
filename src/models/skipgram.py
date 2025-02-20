from typing import Tuple

import torch
from torch import nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGram, self).__init__()
        self.embeddings_input = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_output = nn.Embedding(vocab_size, embedding_dim)

        # initialize embeddings
        initrange = 0.5 / embedding_dim
        self.embeddings_input.weight.data.uniform_(-initrange, initrange)
        self.embeddings_output.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_words, context_words, negative_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        # get center word embeddings
        center_embeds = self.embeddings_input(center_words).squeeze(1)  # [batch_size, embed_dim]

        # get positive context embeddings
        context_embeds = self.embeddings_output(context_words).squeeze(1)  # [batch_size, embed_dim]

        # get negative sample embeddings
        neg_embeds = self.embeddings_output(negative_samples)  # [batch_size, num_neg, embed_dim]

        # compute scores w/ dot product
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, num_neg]

        return pos_score, neg_score
