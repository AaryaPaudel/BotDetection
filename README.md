# Fake Review Detection System

AI Coursework: CU6051NI - Milestone 1

A complete machine learning application for detecting fake (computer-generated) reviews using Natural Language Processing and multiple classification algorithms.

## Features

- **NLP Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature Engineering**: TF-IDF vectorization (5000 features, unigrams + bigrams) + VADER sentiment analysis + normalized ratings
- **Multiple Models**: 
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (LinearSVC)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, and Confusion Matrices

## Project Structure

```
BotDetection/
├── data/
│   └── fake reviews dataset.csv
├── src/
│   ├── preprocessing.py          # Text preprocessing and VADER sentiment
│   ├── feature_engineering.py    # TF-IDF + feature fusion
│   └── models/
│       ├── naive_bayes.py
│       ├── logistic_reg.py
│       └── svm_classifier.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── benchmarking.ipynb
│   └── visualization.ipynb
├── outputs/
│   └── figures/                  # Saved visualizations
├── main.py                       # Main pipeline
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Extract features (TF-IDF + sentiment + rating)
3. Train three classification models
4. Evaluate and display results
5. Generate confusion matrix visualizations

## Dataset

The dataset contains reviews labeled as:
- **CG**: Computer-Generated (Fake)
- **OR**: Original (Real)

## Results

The pipeline generates:
- Classification reports for each model
- Side-by-side confusion matrices
- Performance metrics (Accuracy, Precision, Recall, F1-score)

## Requirements

See `requirements.txt` for full list of dependencies.

