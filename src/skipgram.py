from tqdm import tqdm

from src.common.predict import find_similar_words
from src.common.save_and_load import load_model
from src.common.train import train_handler
from src.common.utils import get_sentences_from_brown_corpus


def main():
    sentences = get_sentences_from_brown_corpus()
    train_handler('skipgram', sentences, embedding_dim=300, window_size=3, num_negative=5, min_count=3, batch_size=256,
                   num_epochs=5, learning_rate=0.001, save_path='skipgram.pt')
    model, vocab, metadata = load_model('skipgram', 'skipgram.pt')

    test_words = ['king', 'queen', 'man', 'woman', 'city']
    for word in tqdm(test_words, desc="Testing similar words"):
        similar_words = find_similar_words(word, vocab.word2idx, model.embeddings_input.weight.detach().cpu())
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")


if __name__ == "__main__":
    main()
