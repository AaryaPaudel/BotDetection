
from sklearn.naive_bayes import MultinomialNB

def train_nb(X_train, y_train):
    model = MultinomialNB(alpha=1.0)
    
    model.fit(X_train, y_train)
    return model