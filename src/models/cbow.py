import torch
from torch import nn


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
        # get context embeddings and average them
        context_embeds = self.embeddings_input(contexts).mean(dim=1)  # [batch_size, embed_dim]

        # get positive target embeddings
        target_embeds = self.embeddings_output(targets).squeeze(1)  # [batch_size, embed_dim]

        # get negative sample embeddings
        neg_embeds = self.embeddings_output(negative_samples)  # [batch_size, num_neg, embed_dim]

        # compute scores w/ dot product
        pos_score = torch.sum(context_embeds * target_embeds, dim=1)  # [batch_size]
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, num_neg]

        return pos_score, neg_score
