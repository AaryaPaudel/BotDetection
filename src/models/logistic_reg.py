from sklearn.linear_model import LogisticRegression

def train_lr(X_train, y_train):
    model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    return model