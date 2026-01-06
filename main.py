import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm

from src.preprocessing import TextPreprocessor
from src.feature_engineering import extract_features
from src.models.naive_bayes import train_nb
from src.models.logistic_reg import train_lr
from src.models.svm_classifier import train_svm


def create_output_directory():
    os.makedirs('outputs/figures', exist_ok=True)


def plot_confusion_matrices(models, X_test, y_test, save_path='outputs/figures/results_summary.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices: Fake Review Detection Models', fontsize=16, fontweight='bold')
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=['CG', 'OR'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['CG (Fake)', 'OR (Real)'],
                   yticklabels=['CG (Fake)', 'OR (Real)'],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved to: {save_path}")
    plt.close()


def run_pipeline():
    print("=" * 70)
    print("FAKE REVIEW DETECTION - MILESTONE 1 PIPELINE")
    print("=" * 70)
    
    # Step 1: Data Loading
    print("\n[Step 1] Loading Labeled Review Dataset...")
    df = pd.read_csv('data/fake reviews dataset.csv')
    print(f"✓ Loaded {len(df)} reviews")
    print(f"  - Labels: {df['label'].value_counts().to_dict()}")
    
    # Step 2: Text Preprocessing & Sentiment Analysis
    print("\n[Step 2] Text Preprocessing & Sentiment Analysis...")
    preprocessor = TextPreprocessor()

    print("  Cleaning text (tokenization, stopword removal, lemmatization)...")
    tqdm.pandas(desc="  Processing text")
    df['cleaned_text'] = df['text_'].progress_apply(preprocessor.clean_text)
    
    print("  Extracting VADER sentiment scores...")
    df['vader_compound'] = df['text_'].progress_apply(preprocessor.get_vader_score)
    print(f"✓ Preprocessed {len(df)} reviews")
    print(f"  - Average sentiment score: {df['vader_compound'].mean():.4f}")
    
    # Step 3: Train-Test Split (before feature engineering to prevent data leakage)
    print("\n[Step 3] Splitting Data (80/20 Stratified Split)...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    y_train = df_train['label']
    y_test = df_test['label']
    print(f"✓ Training set: {len(df_train)} samples")
    print(f"✓ Test set: {len(df_test)} samples")
    print(f"  - Training label distribution: {y_train.value_counts().to_dict()}")
    print(f"  - Test label distribution: {y_test.value_counts().to_dict()}")
    
    # Step 4: Feature Engineering
    print("\n[Step 4] Feature Extraction (TF-IDF + Sentiment + Rating)...")
    print("  Computing TF-IDF features (this may take a few minutes)...")
    with tqdm(total=100, desc="    TF-IDF", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        X_train, X_test, tfidf_vectorizer = extract_features(df_train, df_test=df_test)
        pbar.update(100)
    print(f"✓ Created feature matrices:")
    print(f"  - Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  - Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"  - TF-IDF features: {X_train.shape[1] - 2} (unigrams + bigrams)")
    print(f"  - Additional features: 2 (normalized rating + VADER sentiment)")
    
    # Step 5: Model Training
    print("\n[Step 5] Training Classification Models...")
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Number of training samples: {X_train.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    
    models = {}
    
    # Train Naive Bayes
    print("\n  [5.1] Training Multinomial Naive Bayes...")
    start_time = time.time()
    with tqdm(total=100, desc="    Training", bar_format='{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]') as pbar:
        models["Multinomial Naive Bayes"] = train_nb(X_train, y_train)
        pbar.update(100)
    elapsed = time.time() - start_time
    print(f"  ✓ Naive Bayes completed in {elapsed:.2f} seconds")
    
    # Train Logistic Regression
    print("\n  [5.2] Training Logistic Regression...")
    start_time = time.time()
    with tqdm(total=100, desc="    Training", bar_format='{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]') as pbar:
        models["Logistic Regression"] = train_lr(X_train, y_train)
        pbar.update(100)
    elapsed = time.time() - start_time
    print(f"  ✓ Logistic Regression completed in {elapsed:.2f} seconds")
    
    # Train SVM
    print("\n  [5.3] Training Support Vector Machine (Linear kernel)...")
    start_time = time.time()
    models["Support Vector Machine"] = train_svm(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"  ✓ SVM completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    
    print("\n✓ All models trained successfully")
    
    # Step 6: Model Evaluation
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='CG', average='binary')
        rec = recall_score(y_test, y_pred, pos_label='CG', average='binary')
        f1 = f1_score(y_test, y_pred, pos_label='CG', average='binary')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\n  Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['CG (Fake)', 'OR (Real)']))
    
    # Step 7: Visualization
    print("\n[Step 7] Generating Confusion Matrix Visualizations...")
    create_output_directory()
    plot_confusion_matrices(models, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nSummary of Results:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\n✓ All results saved to: outputs/figures/results_summary.png")


if __name__ == "__main__":
    run_pipeline()