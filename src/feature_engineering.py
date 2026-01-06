from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_features(df_train, df_test=None, text_column='cleaned_text', tfidf_vectorizer=None):
    if tfidf_vectorizer is None:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix_train = tfidf.fit_transform(df_train[text_column])
    else:
        tfidf = tfidf_vectorizer
        tfidf_matrix_train = tfidf.transform(df_train[text_column])
    
    tfidf_array_train = tfidf_matrix_train.toarray()
    
    normalized_rating_train = df_train['rating'].values / 5.0
    
    vader_scores_train = df_train['vader_compound'].values
    normalized_sentiment_train = (vader_scores_train + 1.0) / 2.0
    
    rating_feature_train = normalized_rating_train.reshape(-1, 1)
    sentiment_feature_train = normalized_sentiment_train.reshape(-1, 1)
    
    X_train = np.hstack([tfidf_array_train, rating_feature_train, sentiment_feature_train])
    
    if df_test is not None:
        tfidf_matrix_test = tfidf.transform(df_test[text_column])
        tfidf_array_test = tfidf_matrix_test.toarray()
        
        normalized_rating_test = df_test['rating'].values / 5.0
        vader_scores_test = df_test['vader_compound'].values
        normalized_sentiment_test = (vader_scores_test + 1.0) / 2.0
        
        rating_feature_test = normalized_rating_test.reshape(-1, 1)
        sentiment_feature_test = normalized_sentiment_test.reshape(-1, 1)
        
        X_test = np.hstack([tfidf_array_test, rating_feature_test, sentiment_feature_test])
        return X_train, X_test, tfidf
    
    return X_train, tfidf