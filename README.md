# NLP for Machine Learning

This repository provides a structured and practical overview of **Natural Language Processing (NLP) concepts for Machine Learning**, with a strong focus on **text preprocessing**, **feature representation**, and **word embeddings**. The content is designed as learning notes and explanations that can be used as a reference or study guide.

---

## Overview

In NLP-based machine learning tasks (such as sentiment analysis), raw text must go through several processing steps before it can be used by machine learning algorithms. This repository explains:

1. Text preprocessing techniques
2. Converting text into numerical vectors
3. Classical feature extraction methods
4. Word embeddings and Word2Vec

---

## Text Preprocessing

Text preprocessing is the foundation of any NLP pipeline. It prepares raw text for feature extraction and modeling.

### Common Preprocessing Steps

* Tokenization
* Converting text to lowercase
* Removing punctuation and special characters using regular expressions
* Removing stopwords
* Stemming
* Lemmatization

These steps help normalize text and reduce noise before vectorization.

---

## NLP Pipeline (High Level)

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Text to Vector Conversion
   ↓
Machine Learning Algorithms
```

---

## Text to Vector Representations

Machine learning models require fixed-size numerical input. Therefore, text must be converted into vectors.

### 1. One-Hot Encoding

Each word in the vocabulary is represented as a binary vector.

**Example Documents**:

* D1: the food is good
* D2: the food is bad
* D3: pizza is amazing

**Vocabulary**:

```text
the, food, is, good, bad, pizza, amazing
```

Each word is represented by a vector where only one position is set to 1 and the rest are 0.

#### Advantages

* Easy to understand and implement
* Supported directly in libraries such as scikit-learn and pandas

#### Disadvantages

* Produces sparse matrices
* Leads to high dimensionality and potential overfitting
* No semantic meaning between words
* Out-of-vocabulary (OOV) problem

---

### 2. Bag of Words (BoW)

Bag of Words represents text by counting word frequencies, ignoring word order.

**Example Sentences**:

* S1: He is a good boy
* S2: She is a good girl
* S3: Boy and girl are good

After preprocessing, the vocabulary might be:

```text
good, boy, girl
```

Each sentence is represented by a frequency vector based on this vocabulary.

#### Binary BoW vs Frequency BoW

* Binary BoW: Uses 0 or 1 to indicate word presence
* Frequency BoW: Counts how many times a word appears

#### Advantages

* Simple and intuitive
* Fixed-size input suitable for ML algorithms

#### Disadvantages

* Sparse representation
* Ignores word order
* No semantic meaning
* Out-of-vocabulary issue

---

### 3. N-Grams

N-grams capture local word order by considering sequences of words.

Examples:

* Unigrams: single words
* Bigrams: pairs of words
* Trigrams: sequences of three words

In scikit-learn:

```text
ngram_range=(1,1)  -> Unigrams
ngram_range=(1,2)  -> Unigrams + Bigrams
ngram_range=(1,3)  -> Unigrams + Bigrams + Trigrams
ngram_range=(2,3)  -> Bigrams + Trigrams
```

N-grams partially capture context but still suffer from sparsity.

---

### 4. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF improves upon BoW by weighting words based on their importance.

* Term Frequency (TF):

```text
TF = (Number of times word appears in a document) / (Total words in document)
```

* Inverse Document Frequency (IDF):

```text
IDF = log(Number of documents / Number of documents containing the word)
```

* Final TF-IDF score:

```text
TF-IDF = TF × IDF
```

#### Advantages

* Reduces importance of common words
* Captures word importance
* Fixed-size representation

#### Disadvantages

* Still sparse
* No deep semantic understanding
* Out-of-vocabulary problem remains

---

## Word Embeddings

Word embeddings represent words as dense, real-valued vectors that capture semantic meaning. Words with similar meanings are closer in vector space.

### Types of Embeddings

1. Count-based methods

   * One-Hot Encoding
   * Bag of Words
   * TF-IDF

2. Predictive (Deep Learning-based) methods

   * Word2Vec

     * CBOW (Continuous Bag of Words)
     * Skip-gram

---

## Word2Vec

Word2Vec is a neural network-based technique introduced in 2013 to learn word representations from large text corpora.

Each word is mapped to a fixed-size dense vector (commonly 100–300 dimensions).

### Semantic Relationships

Word embeddings preserve semantic relationships. For example:

```text
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

---

## Cosine Similarity

Cosine similarity measures the similarity between two word vectors by computing the cosine of the angle between them.

```text
Cosine Similarity = (A · B) / (||A|| × ||B||)
Distance = 1 − Cosine Similarity
```

It is commonly used with Word2Vec embeddings to compare word meanings.

---

## Word2Vec Architectures

### 1. CBOW (Continuous Bag of Words)

* Predicts the target word given its surrounding context words
* Faster training
* Performs well on smaller datasets

### 2. Skip-gram

* Predicts surrounding context words given a target word
* Performs better on large datasets
* Captures rare words more effectively

### When to Use

* Small datasets: CBOW
* Large datasets: Skip-gram

---

## Improving Word2Vec Models

* Increase training data size
* Increase context window size
* Train on domain-specific corpora

Google’s Word2Vec model was trained on billions of words and produces 300-dimensional embeddings.

---

## Advantages of Word2Vec

* Dense vector representations (not sparse)
* Captures semantic relationships
* Fixed-size embeddings regardless of vocabulary size
* Handles out-of-vocabulary words better through pretrained models

---

## Conclusion

This repository provides a clear progression from basic text preprocessing to advanced word embeddings. It serves as a strong foundation for tasks such as sentiment analysis, text classification, and other NLP-based machine learning applications.

---

## Author

Hesham El Desoky
Machine Learning Engineer

---

## License

This project is licensed under the MIT License.
