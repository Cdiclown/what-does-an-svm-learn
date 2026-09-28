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


THRESHOLDS = [
    0,
    20,
    50,
    100,
    200,
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


def count_words(text: str) -> int:

    return len(
        text.split()
    )


def main() -> None:

    # --------------------------------------------------
    # DATI
    # --------------------------------------------------

    texts, labels = load_clean_dataset()

    print(
        "Estrazione delle feature linguistiche..."
    )

    all_features = extract_all_features(
        texts
    )

    x_all = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    y_all = np.array(labels)

    lengths = np.array([
        count_words(text)
        for text in texts
    ])

    # --------------------------------------------------
    # CV
    # --------------------------------------------------

    scoring = {
        "accuracy":
            "accuracy",

        "balanced_accuracy":
            "balanced_accuracy",

        "macro_f1":
            "f1_macro",
    }

    summary = []

    print()
    print(
        "=== LENGTH ROBUSTNESS ANALYSIS ==="
    )

    # --------------------------------------------------
    # UNA CV SEPARATA PER OGNI SOGLIA
    # --------------------------------------------------

    for threshold in THRESHOLDS:

        mask = (
            lengths >= threshold
        )

        x = x_all[mask]
        y = y_all[mask]

        num_documents = len(y)

        dementia_count = int(
            np.sum(y == 1)
        )

        control_count = int(
            np.sum(y == 0)
        )

        pipeline = create_pipeline()

        # Ricreiamo la CV per ogni subset.
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        results = cross_validate(
            pipeline,
            x,
            y,
            cv=cv,
            scoring=scoring,
        )

        accuracy_scores = (
            results[
                "test_accuracy"
            ]
        )

        balanced_scores = (
            results[
                "test_balanced_accuracy"
            ]
        )

        macro_f1_scores = (
            results[
                "test_macro_f1"
            ]
        )

        summary.append({
            "threshold":
                threshold,

            "documents":
                num_documents,

            "dementia":
                dementia_count,

            "control":
                control_count,

            "accuracy":
                accuracy_scores.mean(),

            "balanced_accuracy":
                balanced_scores.mean(),

            "macro_f1":
                macro_f1_scores.mean(),

            "macro_f1_std":
                macro_f1_scores.std(),
        })

        print()
        print(
            f"=== >= {threshold} WORDS ==="
        )

        print(
            "Documenti:",
            num_documents,
        )

        print(
            "Dementia:",
            dementia_count,
        )

        print(
            "Control:",
            control_count,
        )

        print(
            "Macro-F1 fold:",
            np.round(
                macro_f1_scores,
                4,
            ),
        )

        print(
            "Macro-F1 mean:",
            round(
                macro_f1_scores.mean(),
                4,
            ),
        )

        print(
            "Macro-F1 std:",
            round(
                macro_f1_scores.std(),
                4,
            ),
        )

        print(
            "Balanced accuracy:",
            round(
                balanced_scores.mean(),
                4,
            ),
        )

    # --------------------------------------------------
    # TABELLA FINALE
    # --------------------------------------------------

    print()
    print(
        "================================"
    )

    print(
        "FINAL LENGTH ROBUSTNESS"
    )

    print(
        "================================"
    )

    print()

    print(
        f"{'Min words':>10s} "
        f"{'N':>6s} "
        f"{'Dem':>6s} "
        f"{'Ctrl':>6s} "
        f"{'Macro-F1':>10s} "
        f"{'Std':>8s} "
        f"{'Bal.Acc':>10s}"
    )

    print(
        "-" * 72
    )

    for item in summary:

        print(
            f"{item['threshold']:10d} "
            f"{item['documents']:6d} "
            f"{item['dementia']:6d} "
            f"{item['control']:6d} "
            f"{item['macro_f1']:10.4f} "
            f"{item['macro_f1_std']:8.4f} "
            f"{item['balanced_accuracy']:10.4f}"
        )


if __name__ == "__main__":
    main()