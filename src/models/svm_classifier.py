from sklearn.svm import LinearSVC

def train_svm(X_train, y_train):
    print(f"      Training on {len(X_train):,} samples (full dataset)...", end="", flush=True)

    model = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42,
        dual=False,
        tol=1e-4,
    )

    model.fit(X_train, y_train)
    print(" ✓", flush=True)
    return model