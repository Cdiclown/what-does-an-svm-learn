import csv
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline import load_clean_dataset
from delexicalize import delexicalize_documents
from linguistic_features import extract_all_features


# --------------------------------------------------
# FEATURE LINGUISTICHE
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


# --------------------------------------------------
# PIPELINE 1 - FULL TF-IDF
# --------------------------------------------------

def create_tfidf_pipeline():

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(),
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


# --------------------------------------------------
# PIPELINE 2 - DELEXICALIZED
# --------------------------------------------------

def create_delex_pipeline():

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=str.split,
                token_pattern=None,
                lowercase=False,
            ),
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


# --------------------------------------------------
# PIPELINE 3 - LINGUISTIC ONLY
# --------------------------------------------------

def create_linguistic_pipeline():

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


# --------------------------------------------------
# FUNZONE PER lIMITARE LA LUNGHEZZA DEL TESTO
# --------------------------------------------------

def clean_snippet(
    text: str,
    max_length: int = 700,
) -> str:

    text = " ".join(
        text.split()
    )

    if len(text) > max_length:
        text = (
            text[:max_length]
            + "..."
        )

    return text


# --------------------------------------------------
# STAMPA DI UN ESEMPIO
# --------------------------------------------------

def print_example(
    index,
    texts,
    labels,
    tfidf_predictions,
    delex_predictions,
    linguistic_predictions,
    tfidf_scores,
    delex_scores,
    linguistic_scores,
):

    print()
    print("=" * 80)

    print(
        "DOCUMENT INDEX:",
        index,
    )

    print(
        "TRUE LABEL:",
        labels[index],
        (
            "dementia"
            if labels[index] == 1
            else "control"
        ),
    )

    print()

    print(
        "TF-IDF:",
        tfidf_predictions[index],
        "CORRECT"
        if tfidf_predictions[index]
        == labels[index]
        else "WRONG",
        "| score:",
        round(
            tfidf_scores[index],
            4,
        ),
    )

    print(
        "DELEX:",
        delex_predictions[index],
        "CORRECT"
        if delex_predictions[index]
        == labels[index]
        else "WRONG",
        "| score:",
        round(
            delex_scores[index],
            4,
        ),
    )

    print(
        "LINGUISTIC:",
        linguistic_predictions[index],
        "CORRECT"
        if linguistic_predictions[index]
        == labels[index]
        else "WRONG",
        "| score:",
        round(
            linguistic_scores[index],
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


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    # ==================================================
    # DATI
    # ==================================================

    texts, labels = load_clean_dataset()

    y = np.array(labels)

    num_documents = len(y)

    print(
        "Documenti:",
        num_documents,
    )

    # ==================================================
    # DE-LESSICALIZZAZIONE
    # ==================================================

    print()
    print(
        "De-lessicalizzazione..."
    )

    delex_texts = (
        delexicalize_documents(
            texts
        )
    )

    # ==================================================
    # FEATURE LINGUISTICHE
    # ==================================================

    print(
        "Estrazione feature linguistiche..."
    )

    all_features = (
        extract_all_features(
            texts
        )
    )

    x_linguistic = np.array([
        [
            features[name]
            for name in FEATURE_NAMES
        ]
        for features in all_features
    ])

    # ==================================================
    # ARRAY PER LE PREDIZIONI OUT-OF-FOLD
    # ==================================================

    tfidf_predictions = np.zeros(
        num_documents,
        dtype=int,
    )

    delex_predictions = np.zeros(
        num_documents,
        dtype=int,
    )

    linguistic_predictions = np.zeros(
        num_documents,
        dtype=int,
    )

    tfidf_scores = np.zeros(
        num_documents,
        dtype=float,
    )

    delex_scores = np.zeros(
        num_documents,
        dtype=float,
    )

    linguistic_scores = np.zeros(
        num_documents,
        dtype=float,
    )

    fold_numbers = np.zeros(
        num_documents,
        dtype=int,
    )

    # ==================================================
    # STESSI IDENTICI FOLD
    # ==================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ==================================================
    # CROSS-VALIDATION MANUALE
    # ==================================================

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(texts, y),
        start=1,
    ):

        print()
        print(
            f"Fold {fold_number}..."
        )

        y_train = y[train_index]
        y_test = y[test_index]

        # ----------------------------------------------
        # TESTI ORIGINALI
        # ----------------------------------------------

        train_texts = [
            texts[i]
            for i in train_index
        ]

        test_texts = [
            texts[i]
            for i in test_index
        ]

        # ----------------------------------------------
        # TESTI DELEXICALIZZATI
        # ----------------------------------------------

        train_delex = [
            delex_texts[i]
            for i in train_index
        ]

        test_delex = [
            delex_texts[i]
            for i in test_index
        ]

        # ----------------------------------------------
        # FEATURE LINGUISTICHE
        # ----------------------------------------------

        x_train_ling = (
            x_linguistic[
                train_index
            ]
        )

        x_test_ling = (
            x_linguistic[
                test_index
            ]
        )

        # ==============================================
        # MODELLO TF-IDF
        # ==============================================

        tfidf_pipeline = (
            create_tfidf_pipeline()
        )

        tfidf_pipeline.fit(
            train_texts,
            y_train,
        )

        tfidf_fold_predictions = (
            tfidf_pipeline.predict(
                test_texts
            )
        )

        tfidf_fold_scores = (
            tfidf_pipeline.decision_function(
                test_texts
            )
        )

        # ==============================================
        # MODELLO DELEX
        # ==============================================

        delex_pipeline = (
            create_delex_pipeline()
        )

        delex_pipeline.fit(
            train_delex,
            y_train,
        )

        delex_fold_predictions = (
            delex_pipeline.predict(
                test_delex
            )
        )

        delex_fold_scores = (
            delex_pipeline.decision_function(
                test_delex
            )
        )

        # ==============================================
        # MODELLO LINGUISTIC
        # ==============================================

        linguistic_pipeline = (
            create_linguistic_pipeline()
        )

        linguistic_pipeline.fit(
            x_train_ling,
            y_train,
        )

        linguistic_fold_predictions = (
            linguistic_pipeline.predict(
                x_test_ling
            )
        )

        linguistic_fold_scores = (
            linguistic_pipeline.decision_function(
                x_test_ling
            )
        )

        # ==============================================
        # SALVATAGGIO OOF
        # ==============================================

        tfidf_predictions[
            test_index
        ] = tfidf_fold_predictions

        delex_predictions[
            test_index
        ] = delex_fold_predictions

        linguistic_predictions[
            test_index
        ] = linguistic_fold_predictions

        tfidf_scores[
            test_index
        ] = tfidf_fold_scores

        delex_scores[
            test_index
        ] = delex_fold_scores

        linguistic_scores[
            test_index
        ] = linguistic_fold_scores

        fold_numbers[
            test_index
        ] = fold_number

        # ==============================================
        # F1 DEL FOLD
        # ==============================================

        print(
            "TF-IDF Macro-F1:",
            round(
                f1_score(
                    y_test,
                    tfidf_fold_predictions,
                    average="macro",
                ),
                4,
            ),
        )

        print(
            "Delex Macro-F1:",
            round(
                f1_score(
                    y_test,
                    delex_fold_predictions,
                    average="macro",
                ),
                4,
            ),
        )

        print(
            "Linguistic Macro-F1:",
            round(
                f1_score(
                    y_test,
                    linguistic_fold_predictions,
                    average="macro",
                ),
                4,
            ),
        )

    # ==================================================
    # RISULTATI TOTALI
    # ==================================================

    print()
    print(
        "================================"
    )
    print(
        "OVERALL OUT-OF-FOLD RESULTS"
    )
    print(
        "================================"
    )

    print()

    print(
        "TF-IDF Macro-F1:",
        round(
            f1_score(
                y,
                tfidf_predictions,
                average="macro",
            ),
            4,
        ),
    )

    print(
        "Delex Macro-F1:",
        round(
            f1_score(
                y,
                delex_predictions,
                average="macro",
            ),
            4,
        ),
    )

    print(
        "Linguistic Macro-F1:",
        round(
            f1_score(
                y,
                linguistic_predictions,
                average="macro",
            ),
            4,
        ),
    )

    # ==================================================
    # CORRETTO / SBAGLIATO PER DOCUMENTO
    # ==================================================

    tfidf_correct = (
        tfidf_predictions == y
    )

    delex_correct = (
        delex_predictions == y
    )

    linguistic_correct = (
        linguistic_predictions == y
    )

    # ==================================================
    # PATTERN
    # ==================================================

    patterns = {
        "ALL_CORRECT":
            (
                tfidf_correct
                & delex_correct
                & linguistic_correct
            ),

        "TFIDF_DELEX_ONLY":
            (
                tfidf_correct
                & delex_correct
                & ~linguistic_correct
            ),

        "TFIDF_ONLY":
            (
                tfidf_correct
                & ~delex_correct
                & ~linguistic_correct
            ),

        "DELEX_LING_ONLY":
            (
                ~tfidf_correct
                & delex_correct
                & linguistic_correct
            ),

        "DELEX_ONLY":
            (
                ~tfidf_correct
                & delex_correct
                & ~linguistic_correct
            ),

        "LINGUISTIC_ONLY":
            (
                ~tfidf_correct
                & ~delex_correct
                & linguistic_correct
            ),

        "TFIDF_LING_ONLY":
            (
                tfidf_correct
                & ~delex_correct
                & linguistic_correct
            ),

        "ALL_WRONG":
            (
                ~tfidf_correct
                & ~delex_correct
                & ~linguistic_correct
            ),
    }

    print()
    print(
        "================================"
    )
    print(
        "ERROR AGREEMENT"
    )
    print(
        "================================"
    )

    print()

    for name, mask in patterns.items():

        print(
            f"{name:20s}",
            int(
                np.sum(mask)
            ),
        )

    # ==================================================
    # ALCUNI CONFRONTI PARTICOLARMENTE IMPORTANTI
    # ==================================================

    print()
    print(
        "TF-IDF correct, Delex wrong:",
        int(
            np.sum(
                tfidf_correct
                & ~delex_correct
            )
        ),
    )

    print(
        "TF-IDF correct, Linguistic wrong:",
        int(
            np.sum(
                tfidf_correct
                & ~linguistic_correct
            )
        ),
    )

    print(
        "TF-IDF wrong, Delex correct:",
        int(
            np.sum(
                ~tfidf_correct
                & delex_correct
            )
        ),
    )

    print(
        "TF-IDF wrong, Linguistic correct:",
        int(
            np.sum(
                ~tfidf_correct
                & linguistic_correct
            )
        ),
    )

    # ==================================================
    # ESEMPI QUALITATIVI
    # ==================================================

    # --------------------------------------------------
    # 1. TF-IDF + DELEX corretti,
    #    linguistic sbagliato
    # --------------------------------------------------

    indices = np.where(
        patterns[
            "TFIDF_DELEX_ONLY"
        ]
    )[0]

    print()
    print()
    print(
        "################################"
    )
    print(
        "TF-IDF + DELEX CORRECT"
    )
    print(
        "LINGUISTIC WRONG"
    )
    print(
        "################################"
    )

    for index in indices[:5]:

        print_example(
            index,
            texts,
            y,
            tfidf_predictions,
            delex_predictions,
            linguistic_predictions,
            tfidf_scores,
            delex_scores,
            linguistic_scores,
        )

    # --------------------------------------------------
    # 2. SOLO TF-IDF corretto
    # --------------------------------------------------

    indices = np.where(
        patterns[
            "TFIDF_ONLY"
        ]
    )[0]

    print()
    print()
    print(
        "################################"
    )
    print(
        "ONLY TF-IDF CORRECT"
    )
    print(
        "################################"
    )

    for index in indices[:5]:

        print_example(
            index,
            texts,
            y,
            tfidf_predictions,
            delex_predictions,
            linguistic_predictions,
            tfidf_scores,
            delex_scores,
            linguistic_scores,
        )

    # --------------------------------------------------
    # 3. TF-IDF sbaglia,
    #    DELEX + LINGUISTIC corretti
    # --------------------------------------------------

    indices = np.where(
        patterns[
            "DELEX_LING_ONLY"
        ]
    )[0]

    print()
    print()
    print(
        "################################"
    )
    print(
        "TF-IDF WRONG"
    )
    print(
        "DELEX + LINGUISTIC CORRECT"
    )
    print(
        "################################"
    )

    for index in indices[:5]:

        print_example(
            index,
            texts,
            y,
            tfidf_predictions,
            delex_predictions,
            linguistic_predictions,
            tfidf_scores,
            delex_scores,
            linguistic_scores,
        )

    # --------------------------------------------------
    # 4. TUTTI SBAGLIANO
    # --------------------------------------------------

    indices = np.where(
        patterns[
            "ALL_WRONG"
        ]
    )[0]

    print()
    print()
    print(
        "################################"
    )
    print(
        "ALL MODELS WRONG"
    )
    print(
        "################################"
    )

    for index in indices[:5]:

        print_example(
            index,
            texts,
            y,
            tfidf_predictions,
            delex_predictions,
            linguistic_predictions,
            tfidf_scores,
            delex_scores,
            linguistic_scores,
        )

    # ==================================================
    # SALVATAGGIO CSV
    # ==================================================

    output_file = (
        "model_comparison_errors.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "document_index",
            "fold",
            "true_label",

            "tfidf_prediction",
            "tfidf_correct",
            "tfidf_score",

            "delex_prediction",
            "delex_correct",
            "delex_score",

            "linguistic_prediction",
            "linguistic_correct",
            "linguistic_score",

            "text",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            num_documents
        ):

            writer.writerow({
                "document_index":
                    index,

                "fold":
                    fold_numbers[index],

                "true_label":
                    y[index],

                "tfidf_prediction":
                    tfidf_predictions[index],

                "tfidf_correct":
                    int(
                        tfidf_correct[index]
                    ),

                "tfidf_score":
                    tfidf_scores[index],

                "delex_prediction":
                    delex_predictions[index],

                "delex_correct":
                    int(
                        delex_correct[index]
                    ),

                "delex_score":
                    delex_scores[index],

                "linguistic_prediction":
                    linguistic_predictions[index],

                "linguistic_correct":
                    int(
                        linguistic_correct[
                            index
                        ]
                    ),

                "linguistic_score":
                    linguistic_scores[index],

                "text":
                    texts[index],
            })

    print()
    print(
        "File salvato:",
        output_file,
    )


if __name__ == "__main__":
    main()