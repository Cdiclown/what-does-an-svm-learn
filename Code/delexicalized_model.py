import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from baseline import load_clean_dataset
from delexicalize import delexicalize_documents

def main() -> None:

    texts, labels = load_clean_dataset()

    print("De-lessicalizzazione dei documenti...")

    delex_texts = delexicalize_documents(texts)

    print("Documenti trasformati:", len(delex_texts))

    print()
    print("Esempio originale:")
    print(texts[0][:500])

    print()
    print("Esempio de-lessicalizzato:")
    print(delex_texts[0][:500])

    delex_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=str.split,
                token_pattern=None,
                lowercase=False,
            )
        ),
        (
            "svm",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
            )
        ),
    ])
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    results = cross_validate(
        delex_pipeline,
        delex_texts,
        labels,
        cv=cv,
        scoring=scoring,
    )
    print()
    print("=== DE-LEXICALIZED MODEL ===")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        scores = results[f"test_{metric}"]

        print()
        print(metric)
        print("Fold:", np.round(scores, 4))
        print("Media:", round(scores.mean(), 4))
        print("Std:", round(scores.std(), 4))
if __name__ == "__main__":
    main()
