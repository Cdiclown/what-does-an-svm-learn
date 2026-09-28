import csv
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline import load_clean_dataset
from linguistic_features import extract_all_features


# --------------------------------------------------
# FEATURE USATE DAL MODELLO
# --------------------------------------------------

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


# Per non stampare 22 valori per ogni esempio,
# mostriamo queste feature principali.
DISPLAY_FEATURES = [
    "function_word_rate",
    "noun_rate",
    "past_verb_rate",
    "mattr",
    "pronoun_rate",
    "avg_dependency_depth",
    "connective_rate",
    "adjacent_sentence_overlap",
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


def clean_snippet(
    text: str,
    max_length: int = 700,
) -> str:
    """
    Rende il testo più leggibile
    nel terminale.
    """

    text = " ".join(
        text.split()
    )

    if len(text) > max_length:
        text = (
            text[:max_length]
            + "..."
        )

    return text


def print_example(
    index,
    texts,
    labels,
    predictions,
    decision_scores,
    fold_numbers,
    all_features,
):

    true_label = labels[index]
    predicted_label = predictions[index]

    print()
    print("=" * 75)

    print(
        "DOCUMENT INDEX:",
        index,
    )

    print(
        "FOLD:",
        fold_numbers[index],
    )

    print(
        "TRUE LABEL:",
        true_label,
        (
            "dementia"
            if true_label == 1
            else "control"
        ),
    )

    print(
        "PREDICTED:",
        predicted_label,
        (
            "dementia"
            if predicted_label == 1
            else "control"
        ),
    )

    print(
        "DECISION SCORE:",
        round(
            decision_scores[index],
            4,
        ),
    )

    print()
    print("FEATURES:")

    for feature_name in DISPLAY_FEATURES:

        print(
            f"  {feature_name:30s}",
            round(
                all_features[index][
                    feature_name
                ],
                4,
            ),
        )

    print()
    print("TEXT:")

    print(
        clean_snippet(
            texts[index]
        )
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

    x = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    y = np.array(labels)

    num_documents = len(y)

    print(
        "Documenti:",
        num_documents,
    )

    # --------------------------------------------------
    # ARRAY PER LE PREDIZIONI OUT-OF-FOLD
    # --------------------------------------------------

    # Ogni documento comparirà esattamente
    # una volta nel test set di un fold.

    predictions = np.zeros(
        num_documents,
        dtype=int,
    )

    decision_scores = np.zeros(
        num_documents,
        dtype=float,
    )

    fold_numbers = np.zeros(
        num_documents,
        dtype=int,
    )

    # --------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

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

        fold_predictions = (
            pipeline.predict(
                x_test
            )
        )

        # Positive score -> dementia
        # Negative score -> control
        fold_scores = (
            pipeline.decision_function(
                x_test
            )
        )

        # Rimettiamo ogni risultato
        # nella posizione originale
        # del documento.
        predictions[test_index] = (
            fold_predictions
        )

        decision_scores[test_index] = (
            fold_scores
        )

        fold_numbers[test_index] = (
            fold_number
        )

        fold_f1 = f1_score(
            y_test,
            fold_predictions,
            average="macro",
        )

        print(
            f"Fold {fold_number} "
            f"- Macro-F1: "
            f"{fold_f1:.4f}"
        )

    # --------------------------------------------------
    # RISULTATI GLOBALI OOF
    # --------------------------------------------------

    overall_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    print()
    print(
        "=== OUT-OF-FOLD RESULTS ==="
    )

    print(
        "Macro-F1:",
        round(
            overall_f1,
            4,
        ),
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(matrix)

    # --------------------------------------------------
    # TROVIAMO I QUATTRO TIPI DI CASO
    # --------------------------------------------------

    false_positive_indices = np.where(
        (y == 0)
        & (predictions == 1)
    )[0]

    false_negative_indices = np.where(
        (y == 1)
        & (predictions == 0)
    )[0]

    true_positive_indices = np.where(
        (y == 1)
        & (predictions == 1)
    )[0]

    true_negative_indices = np.where(
        (y == 0)
        & (predictions == 0)
    )[0]

    print()
    print(
        "False positives:",
        len(false_positive_indices),
    )

    print(
        "False negatives:",
        len(false_negative_indices),
    )

    print(
        "True positives:",
        len(true_positive_indices),
    )

    print(
        "True negatives:",
        len(true_negative_indices),
    )

    # --------------------------------------------------
    # ERRORI PIÙ SICURI
    # --------------------------------------------------

    # FALSE POSITIVE:
    # veri control che la SVM considera
    # fortemente dementia.
    false_positive_indices = sorted(
        false_positive_indices,
        key=lambda i:
            decision_scores[i],
        reverse=True,
    )

    # FALSE NEGATIVE:
    # veri dementia con score
    # molto negativo.
    false_negative_indices = sorted(
        false_negative_indices,
        key=lambda i:
            decision_scores[i],
    )

    # TRUE POSITIVE molto sicuri.
    true_positive_indices = sorted(
        true_positive_indices,
        key=lambda i:
            decision_scores[i],
        reverse=True,
    )

    # TRUE NEGATIVE molto sicuri.
    true_negative_indices = sorted(
        true_negative_indices,
        key=lambda i:
            decision_scores[i],
    )

    # --------------------------------------------------
    # STAMPA 5 FALSE POSITIVE
    # --------------------------------------------------

    print()
    print()
    print(
        "################################"
    )
    print(
        "TOP 5 FALSE POSITIVES"
    )
    print(
        "True = control, predicted = dementia"
    )
    print(
        "################################"
    )

    for index in false_positive_indices[:5]:

        print_example(
            index,
            texts,
            y,
            predictions,
            decision_scores,
            fold_numbers,
            all_features,
        )

    # --------------------------------------------------
    # STAMPA 5 FALSE NEGATIVE
    # --------------------------------------------------

    print()
    print()
    print(
        "################################"
    )
    print(
        "TOP 5 FALSE NEGATIVES"
    )
    print(
        "True = dementia, predicted = control"
    )
    print(
        "################################"
    )

    for index in false_negative_indices[:5]:

        print_example(
            index,
            texts,
            y,
            predictions,
            decision_scores,
            fold_numbers,
            all_features,
        )

    # --------------------------------------------------
    # 5 CORRETTI AD ALTA CONFIDENZA
    # --------------------------------------------------

    # Prendiamo alcuni casi chiarissimi
    # di entrambe le classi.

    print()
    print()
    print(
        "################################"
    )
    print(
        "HIGH-CONFIDENCE CORRECT CASES"
    )
    print(
        "################################"
    )

    print()
    print(
        "--- DEMENTIA ---"
    )

    for index in true_positive_indices[:3]:

        print_example(
            index,
            texts,
            y,
            predictions,
            decision_scores,
            fold_numbers,
            all_features,
        )

    print()
    print(
        "--- CONTROL ---"
    )

    for index in true_negative_indices[:3]:

        print_example(
            index,
            texts,
            y,
            predictions,
            decision_scores,
            fold_numbers,
            all_features,
        )

    # --------------------------------------------------
    # SALVIAMO TUTTO IN CSV
    # --------------------------------------------------

    output_file = (
        "linguistic_error_analysis.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        field_names = [
            "document_index",
            "fold",
            "true_label",
            "predicted_label",
            "decision_score",
            "correct",
        ]

        field_names += FEATURE_NAMES

        field_names += [
            "text"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()

        for index in range(
            num_documents
        ):

            row = {
                "document_index":
                    index,

                "fold":
                    fold_numbers[index],

                "true_label":
                    y[index],

                "predicted_label":
                    predictions[index],

                "decision_score":
                    decision_scores[index],

                "correct":
                    int(
                        y[index]
                        == predictions[index]
                    ),

                "text":
                    texts[index],
            }

            for feature_name in (
                FEATURE_NAMES
            ):

                row[feature_name] = (
                    all_features[index][
                        feature_name
                    ]
                )

            writer.writerow(row)

    print()
    print(
        "File salvato:",
        output_file,
    )


if __name__ == "__main__":
    main()