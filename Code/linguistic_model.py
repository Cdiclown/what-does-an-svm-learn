import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline import load_clean_dataset
from linguistic_features import extract_all_features


FEATURE_NAMES = [
    "avg_sentence_length",
    "sentence_length_std",
    "avg_dependency_depth",
    "subordination_rate",
    "clause_to_sentence_ratio",
    "avg_word_length",
    "mattr",
    "repetition_rate",
    "pronoun_rate",
    "noun_pronoun_ratio",
    "noun_rate",
    "verb_rate",
    "adjective_rate",
    "adverb_rate",
    "function_word_rate",
    "vague_word_rate",
    "adjacent_sentence_overlap",
    "connective_rate",
    "hedge_rate",
    "modal_verb_rate",
    "past_verb_rate",
    "present_verb_rate",
]


def main() -> None:

    # -----------------------------------------------
    # CARICAMENTO DATI
    # -----------------------------------------------

    texts, labels = load_clean_dataset()

    print("Estrazione delle feature linguistiche...")

    all_features = extract_all_features(texts)

    print(
        "Documenti:",
        len(all_features),
    )

    # -----------------------------------------------
    # COSTRUZIONE DELLA MATRICE X
    # -----------------------------------------------

    # Ogni riga = un documento
    # Ogni colonna = una feature linguistica

    x = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    y = np.array(labels)

    print(
        "Forma della matrice:",
        x.shape,
    )

    # Dovrebbe essere:
    # (3644, 22)

    # -----------------------------------------------
    # PIPELINE
    # -----------------------------------------------

    linguistic_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # -----------------------------------------------
    # CROSS-VALIDATION
    # -----------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy":
            "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    results = cross_validate(
        linguistic_pipeline,
        x,
        y,
        cv=cv,
        scoring=scoring,
    )

    # -----------------------------------------------
    # RISULTATI
    # -----------------------------------------------

    print()
    print(
        "=== LINGUISTIC FEATURES ONLY ==="
    )

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:

        scores = results[
            f"test_{metric}"
        ]

        print()
        print(metric)

        print(
            "Fold:",
            np.round(scores, 4),
        )

        print(
            "Media:",
            round(
                scores.mean(),
                4,
            ),
        )

        print(
            "Std:",
            round(
                scores.std(),
                4,
            ),
        )


if __name__ == "__main__":
    main()