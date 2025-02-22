import argparse
import os.path

from tqdm import tqdm

from src.common.predict import find_similar_words
from src.common.utils import get_sentences_from_brown_corpus
from src.models.svd import WordEmbeddingSVD


def main(path_to_we: str):
    sentences = get_sentences_from_brown_corpus()
    if path_to_we and os.path.exists(path_to_we):
        print("Loading pretrained word embeddings.")
        model = WordEmbeddingSVD.load(path_to_we)
    else:
        print("Training model as pretrained word embeddings do not exist.")
        model = WordEmbeddingSVD(window_size=3, min_freq=3, embedding_dim=300)
        model.train(sentences)
        model.save('svd.pt')


    test_words = ['king', 'queen', 'man', 'woman', 'city']
    for word in tqdm(test_words, desc="Testing similar words"):
        similar_words = find_similar_words(word, model.word2idx, model.embeddings.detach().cpu())
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="Path to the pretrained word embeddings", required=False, default=None, type=str)
    args = parser.parse_args()
    main(args.e)
