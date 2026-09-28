import numpy as np

from sklearn.model_selection import StratifiedKFold
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


def create_pipeline():

    return Pipeline([
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


def main() -> None:

    # --------------------------------------------------
    # CARICAMENTO DATI
    # --------------------------------------------------

    texts, labels = load_clean_dataset()

    print("Estrazione delle feature linguistiche...")

    all_features = extract_all_features(texts)

    # Ogni riga = documento
    # Ogni colonna = feature linguistica
    x = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    y = np.array(labels)

    print(
        "Forma matrice:",
        x.shape,
    )

    # --------------------------------------------------
    # STESSA 5-FOLD CV USATA PRIMA
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Qui salveremo:
    #
    # fold 1 -> 22 coefficienti
    # fold 2 -> 22 coefficienti
    # ...
    # fold 5 -> 22 coefficienti

    fold_coefficients = []

    # --------------------------------------------------
    # TRAINING NEI 5 FOLD
    # --------------------------------------------------

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(x, y),
        start=1,
    ):

        x_train = x[train_index]
        y_train = y[train_index]

        pipeline = create_pipeline()

        # Fit solo sul training fold.
        pipeline.fit(
            x_train,
            y_train,
        )

        # Recuperiamo la SVM già addestrata.
        svm = pipeline.named_steps["svm"]

        # Per classificazione binaria:
        # un coefficiente per ciascuna feature.
        coefficients = svm.coef_[0]

        fold_coefficients.append(
            coefficients
        )

        print(
            f"Fold {fold_number} completato"
        )

    # --------------------------------------------------
    # MATRICE 5 x 22
    # --------------------------------------------------

    fold_coefficients = np.array(
        fold_coefficients
    )

    print()
    print(
        "Forma matrice coefficienti:",
        fold_coefficients.shape,
    )

    # --------------------------------------------------
    # MEDIA E DEVIAZIONE STANDARD
    # --------------------------------------------------

    mean_coefficients = np.mean(
        fold_coefficients,
        axis=0,
    )

    std_coefficients = np.std(
        fold_coefficients,
        axis=0,
    )

    # --------------------------------------------------
    # ORDINIAMO PER IMPORTANZA ASSOLUTA
    # --------------------------------------------------

    sorted_indices = np.argsort(
        np.abs(mean_coefficients)
    )[::-1]

    # --------------------------------------------------
    # RISULTATI
    # --------------------------------------------------

    print()
    print(
        "=== FEATURE IMPORTANCE ==="
    )

    print()
    print(
        f"{'Feature':32s} "
        f"{'Mean coef':>10s} "
        f"{'Std':>8s} "
        f"{'Direction':>12s}"
    )

    print("-" * 68)

    for index in sorted_indices:

        feature_name = (
            FEATURE_NAMES[index]
        )

        mean_coef = (
            mean_coefficients[index]
        )

        std_coef = (
            std_coefficients[index]
        )

        if mean_coef > 0:
            direction = "dementia"
        else:
            direction = "control"

        print(
            f"{feature_name:32s} "
            f"{mean_coef:10.4f} "
            f"{std_coef:8.4f} "
            f"{direction:>12s}"
        )

    # --------------------------------------------------
    # COEFFICIENTI DEI SINGOLI FOLD
    # --------------------------------------------------

    print()
    print(
        "=== COEFFICIENTI PER FOLD ==="
    )

    for index in sorted_indices:

        print()
        print(
            FEATURE_NAMES[index]
        )

        print(
            np.round(
                fold_coefficients[
                    :,
                    index
                ],
                4,
            )
        )


if __name__ == "__main__":
    main()