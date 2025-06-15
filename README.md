
# Genre Classifier

This project is a **Genre Classifier** that uses various machine learning techniques to predict the genre of a song based on its lyrics. The classifier supports genres such as **Pop**, **Rap**, **Gospel**, and more. It implements multiple approaches, including **k-Nearest Neighbors (k-NN)**, **TF-IDF vectorization**, **k-Means Clustering**, and **FastText embeddings**.

## Features

- **k-Nearest Neighbors (k-NN):**
  - Predicts the genre of a song by finding the most similar songs in the dataset.
  - Uses cosine similarity and TF-IDF vectorization for accurate predictions.

- **k-Means Clustering:**
  - Groups songs into clusters based on their lyrical content.
  - Identifies the most frequent words in each cluster.

- **FastText Embeddings:**
  - Leverages deep learning-based word embeddings for more nuanced vector representations of lyrics.
  - Provides improved accuracy for genre prediction.

- **Custom Song Class:**
  - Encapsulates song lyrics and provides methods for vectorization, magnitude calculation, and similarity computation.

## Project Structure





### Key Files

- **`test1.py`:**
  - Implements a basic k-NN classifier using cosine similarity and unit vectors.
  - Focuses on genre prediction using simple vectorization techniques.

- **`test2.py`:**
  - Enhances the k-NN classifier with TF-IDF vectorization to downweight common words.
  - Includes k-Means clustering to group songs and identify frequent words in clusters.

- **`test3.py`:**
  - Introduces FastText embeddings for deep vectorization of lyrics.
  - Uses FastText-based k-NN and k-Means clustering for genre prediction.

- **`test4.py`:**
  - Combines multiple approaches and evaluates their performance.
  - Provides a comprehensive analysis of genre prediction accuracy.

## How It Works

1. **Data Preparation:**
   - Lyrics are loaded from text files in the `kNN-data` directory, organized by genre.

2. **Vectorization:**
   - Lyrics are preprocessed by removing punctuation and stopwords.
   - TF-IDF or FastText embeddings are used to convert lyrics into numerical vectors.

3. **Genre Prediction:**
   - For k-NN, the classifier finds the `k` most similar songs and predicts the genre based on majority voting.
   - For k-Means, songs are grouped into clusters, and the most frequent words in each cluster are identified.

4. **Evaluation:**
   - The classifier outputs the predicted genre and provides insights into the clustering process.

## Usage

1. Place your song lyrics in the `newinput.txt` file.
2. Run one of the test scripts to predict the genre:
   - For k-NN with TF-IDF:
     ```bash
     python test2.py
     ```
   - For FastText-based predictions:
     ```bash
     python test3.py
     ```
3. Follow the prompts to view the predicted genre and clustering results.
