import argparse
import os.path

from tqdm import tqdm

from src.common.predict import find_similar_words
from src.common.save_and_load import load_model
from src.common.train import train_handler
from src.common.utils import get_sentences_from_brown_corpus


def main(path_to_we: str):
    sentences = get_sentences_from_brown_corpus()
    if path_to_we and os.path.exists(path_to_we):
        print("Loading pretrained word embeddings.")
        model, vocab, metadata = load_model('cbow', path_to_we)
    else:
        print("Training model as pretrained word embeddings do not exist.")
        model, vocab = train_handler('cbow', sentences, embedding_dim=300, window_size=3, num_negative=5, min_count=3,
                                     batch_size=256, num_epochs=10, learning_rate=0.001, save_path='cbow.pt')

    test_words = ['king', 'queen', 'man', 'woman', 'city']
    for word in tqdm(test_words, desc="Testing similar words"):
        similar_words = find_similar_words(word, vocab.word2idx, model.embeddings_input.weight.detach().cpu())
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="Path to the pretrained word embeddings", required=False, default=None, type=str)
    args = parser.parse_args()
    main(args.e)
