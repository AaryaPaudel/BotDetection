import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        self.analyzer = SentimentIntensityAnalyzer()
        
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text_lower = text.lower()
        
        tokens = word_tokenize(text_lower)
        
        cleaned_tokens = []
        for w in tokens:
            if w.isalpha() and w not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(w)
                cleaned_tokens.append(lemmatized)
        
        return " ".join(cleaned_tokens)

    def get_vader_score(self, text):
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']