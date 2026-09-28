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


# --------------------------------------------------
# GRUPPI DI FEATURE
# --------------------------------------------------

FEATURE_GROUPS = {

    "ALL_22": [
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
    ],

    "LEXICAL": [
        "avg_word_length",
        "mattr",
        "repetition_rate",
        "vague_word_rate",
    ],

    "SYNTACTIC": [
        "avg_sentence_length",
        "sentence_length_std",
        "avg_dependency_depth",
        "subordination_rate",
        "clause_to_sentence_ratio",
    ],

    "POS_REFERENCE": [
        "pronoun_rate",
        "noun_pronoun_ratio",
        "noun_rate",
        "verb_rate",
        "adjective_rate",
        "adverb_rate",
        "function_word_rate",
    ],

    "DISCOURSE": [
        "adjacent_sentence_overlap",
        "connective_rate",
        "hedge_rate",
    ],

    "TENSE_MODAL": [
        "modal_verb_rate",
        "past_verb_rate",
        "present_verb_rate",
    ],
}


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


def build_matrix(
    all_features,
    feature_names,
):
    """
    Costruisce la matrice X usando
    soltanto le feature del gruppo scelto.
    """

    return np.array([
        [
            features[name]
            for name in feature_names
        ]
        for features in all_features
    ])


def main() -> None:

    # --------------------------------------------------
    # CARICAMENTO DATI
    # --------------------------------------------------

    texts, labels = load_clean_dataset()

    print(
        "Estrazione delle feature linguistiche..."
    )

    all_features = extract_all_features(
        texts
    )

    y = np.array(labels)

    print(
        "Documenti:",
        len(all_features),
    )

    # --------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------

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

    # Salviamo qui i risultati finali.
    summary = []

    print()
    print(
        "=== FEATURE GROUP ANALYSIS ==="
    )

    # --------------------------------------------------
    # UN MODELLO PER OGNI GRUPPO
    # --------------------------------------------------

    for (
        group_name,
        feature_names,
    ) in FEATURE_GROUPS.items():

        x = build_matrix(
            all_features,
            feature_names,
        )

        pipeline = create_pipeline()

        results = cross_validate(
            pipeline,
            x,
            y,
            cv=cv,
            scoring=scoring,
        )

        accuracy_scores = (
            results["test_accuracy"]
        )

        balanced_scores = (
            results[
                "test_balanced_accuracy"
            ]
        )

        macro_f1_scores = (
            results["test_macro_f1"]
        )

        summary.append({
            "group": group_name,
            "num_features":
                len(feature_names),

            "accuracy_mean":
                accuracy_scores.mean(),

            "accuracy_std":
                accuracy_scores.std(),

            "balanced_mean":
                balanced_scores.mean(),

            "balanced_std":
                balanced_scores.std(),

            "macro_f1_mean":
                macro_f1_scores.mean(),

            "macro_f1_std":
                macro_f1_scores.std(),
        })

        print()
        print(
            f"=== {group_name} ==="
        )

        print(
            "Feature:",
            len(feature_names),
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
    # RANKING FINALE
    # --------------------------------------------------

    summary.sort(
        key=lambda item:
            item["macro_f1_mean"],
        reverse=True,
    )

    print()
    print(
        "================================"
    )

    print(
        "FINAL RANKING"
    )

    print(
        "================================"
    )

    print()

    print(
        f"{'Group':20s} "
        f"{'N':>4s} "
        f"{'Macro-F1':>10s} "
        f"{'Std':>8s} "
        f"{'Bal.Acc':>10s}"
    )

    print(
        "-" * 58
    )

    for item in summary:

        print(
            f"{item['group']:20s} "
            f"{item['num_features']:4d} "
            f"{item['macro_f1_mean']:10.4f} "
            f"{item['macro_f1_std']:8.4f} "
            f"{item['balanced_mean']:10.4f}"
        )


if __name__ == "__main__":
    main()