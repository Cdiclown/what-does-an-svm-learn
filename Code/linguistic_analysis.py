import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
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
    # DATI
    # --------------------------------------------------

    texts, labels = load_clean_dataset()

    print("Estrazione delle feature linguistiche...")

    all_features = extract_all_features(texts)

    x = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    y = np.array(labels)

    print("Forma matrice:", x.shape)

    # ==================================================
    # PARTE 1 - CORRELAZIONI
    # ==================================================

    print()
    print("================================")
    print("CORRELATION ANALYSIS")
    print("================================")

    correlation_matrix = np.corrcoef(
        x,
        rowvar=False,
    )

    correlated_pairs = []

    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):

            correlation = (
                correlation_matrix[i, j]
            )

            correlated_pairs.append(
                (
                    abs(correlation),
                    correlation,
                    FEATURE_NAMES[i],
                    FEATURE_NAMES[j],
                )
            )

    # Ordiniamo dalla correlazione assoluta
    # più alta alla più bassa.
    correlated_pairs.sort(
        reverse=True
    )

    print()
    print("=== TOP CORRELATIONS ===")

    for (
        abs_correlation,
        correlation,
        feature_1,
        feature_2,
    ) in correlated_pairs[:20]:

        print(
            f"{feature_1:30s} "
            f"<-> "
            f"{feature_2:30s} "
            f"r = {correlation:7.4f}"
        )

    # --------------------------------------------------
    # Correlazioni forti
    # --------------------------------------------------

    print()
    print(
        "=== STRONG CORRELATIONS |r| >= 0.70 ==="
    )

    strong_found = False

    for (
        abs_correlation,
        correlation,
        feature_1,
        feature_2,
    ) in correlated_pairs:

        if abs_correlation < 0.70:
            break

        strong_found = True

        print(
            f"{feature_1:30s} "
            f"<-> "
            f"{feature_2:30s} "
            f"r = {correlation:7.4f}"
        )

    if not strong_found:
        print(
            "Nessuna correlazione >= 0.70"
        )

    # ==================================================
    # PARTE 2 - PERMUTATION IMPORTANCE
    # ==================================================

    print()
    print("================================")
    print("PERMUTATION IMPORTANCE")
    print("================================")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_importances = []
    baseline_scores = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(x, y),
        start=1,
    ):

        x_train = x[train_index]
        x_test = x[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        pipeline = create_pipeline()

        pipeline.fit(
            x_train,
            y_train,
        )

        # -----------------------------------------------
        # Macro-F1 normale del fold
        # -----------------------------------------------

        predictions = pipeline.predict(
            x_test
        )

        baseline_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        baseline_scores.append(
            baseline_f1
        )

        # -----------------------------------------------
        # Permutation importance
        # -----------------------------------------------

        importance = permutation_importance(
            pipeline,
            x_test,
            y_test,
            scoring="f1_macro",
            n_repeats=20,
            random_state=42 + fold_number,
        )

        fold_importances.append(
            importance.importances_mean
        )

        print(
            f"Fold {fold_number} completato "
            f"- Macro-F1: "
            f"{baseline_f1:.4f}"
        )

    # --------------------------------------------------
    # 5 fold x 22 feature
    # --------------------------------------------------

    fold_importances = np.array(
        fold_importances
    )

    baseline_scores = np.array(
        baseline_scores
    )

    print()
    print(
        "Macro-F1 medio:",
        round(
            baseline_scores.mean(),
            4,
        ),
    )

    print(
        "Macro-F1 std:",
        round(
            baseline_scores.std(),
            4,
        ),
    )

    # --------------------------------------------------
    # IMPORTANZA MEDIA TRA I FOLD
    # --------------------------------------------------

    mean_importances = np.mean(
        fold_importances,
        axis=0,
    )

    std_importances = np.std(
        fold_importances,
        axis=0,
    )

    sorted_indices = np.argsort(
        mean_importances
    )[::-1]

    print()
    print(
        "=== PERMUTATION IMPORTANCE RANKING ==="
    )

    print()

    print(
        f"{'Feature':32s} "
        f"{'Mean drop F1':>14s} "
        f"{'Std':>10s}"
    )

    print("-" * 60)

    for index in sorted_indices:

        print(
            f"{FEATURE_NAMES[index]:32s} "
            f"{mean_importances[index]:14.4f} "
            f"{std_importances[index]:10.4f}"
        )

    # --------------------------------------------------
    # IMPORTANZA NEI SINGOLI FOLD
    # --------------------------------------------------

    print()
    print(
        "=== IMPORTANCE PER FOLD ==="
    )

    for index in sorted_indices:

        print()
        print(
            FEATURE_NAMES[index]
        )

        print(
            np.round(
                fold_importances[
                    :,
                    index
                ],
                4,
            )
        )


if __name__ == "__main__":
    main()
    