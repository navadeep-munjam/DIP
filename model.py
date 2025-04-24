from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import numpy as np

def train_model(X, y, pca_components=2):
    # Safely reduce PCA components based on available data
    max_pca = min(pca_components, X.shape[0], X.shape[1])
    
    # Safe CV folds based on smallest class size
    class_counts = np.bincount(y)
    cv_folds = min(3, class_counts.min()) if class_counts.min() > 1 else 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("pca", PCA(n_components=max_pca)),
        ("svm", SVC(probability=True))
    ])

    params = {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ['scale', 'auto'],
        "svm__kernel": ['rbf'],
        "svm__class_weight": ['balanced']
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=cv_folds,
        verbose=1,
        n_jobs=-1,
        error_score='raise'  # to help you debug quickly
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("\nClassification report on test set:")
    print(classification_report(y_test, grid.predict(X_test)))

    return grid

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)

def load_model(path='model.joblib'):
    model = joblib.load(path)
    print("✅ Model loaded from", path)
    return model
