# Static Word Embeddings

Static Word Embedding models, such as Singular Value Decomposition (SVD) on a Co-Occurrence Matrix, Continuous Bag of
Words (CBOW) with Negative Sampling and Skipgram with Negative Sampling. Also includes a way to evaluate the quality of
the word embeddings produced, by comparing the cosine similarities between pairs of words that have been assigned a
human value in the WordSim353 (Crowd) dataset using the Spearman Rank Correlation method.

Implemented in Python using PyTorch. This corresponds to Assignment-3 of the Introduction to Natural Language Processing
course at IIIT Hyderabad, taken in the Spring'25 semester.

---

# Pre-requisites

1. `python 3.12`
2. A python package manager such as `pip` or `conda`.
3. [pytorch](https://pytorch.org/get-started/locally/)
4. (OPTIONAL) `virtualenv` to create a virtual environment.
5. All the python libraries mentioned in `requirements.txt`.
6. [WordSim353 (Crowd)](https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd)

---

# Word Embedding Generation

## Instructions to run

### SVD

```bash
python3 -m src.svd
```

### Continuous Bag of Words

```bash
python3 -m src.cbow
```

### Skipgram

```bash
python3 -m src.skipgram
```

---

# Word Similarity Task

> [!NOTE]
> The dataset and the columns on which this task will be performed are all HARD-CODED in [wordsim.py](src/wordsim.py).
> Please make all the necessary changes in order to use a different dataset for this task.

## Instructions to run

```bash
python3 -m src.wordsim <path_to_word_embeddings>
```

---

# Analysis

Please refer to the [report](Report.md) for an analysis of these models.

---

