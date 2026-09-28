import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from baseline import load_clean_dataset


BLOG_ARTIFACTS = {
    "joe",
    "alan",
    "vince",
    "jim",
    "fisher",
    "rodney",
    "atlanta",
    "georgia",
    "athens",
    "missouri",
    "sedalia",
    "blogspot",
    "copyright",
    "http",
    "com",
    "org",
    "facebook",
    "helpparentsagewell",
}

DISCOURSE_ROLE_WORDS = {
    "mom",
    "mother",
    "dad",
    "father",
    "parents",
    "parent",
    "wife",
    "husband",
    "son",
    "daughter",
    "kids",
    "child",
    "children",

    "she",
    "her",
    "hers",
    "he",
    "him",
    "his",

    "me",
    "my",
    "mine",
    "ours",
}
CLINICAL_TOPIC_WORDS = {
    "alzheimer",
    "alzheimers",
    "dementia",
    "ad",
    "lbd",
    "lewy",
    "earlyonset",
    "onset",
    "aging",
    "disease",
    "diagnosis",
    "diagnosed",
    "patient",
    "patients",
    "caregiver",
    "caregivers",
    "caregiving",
    "memory",
    "cognitive",
    "neurologist",
    "neurologists",
    "namenda",
    "aricept",
    "razadyne",
    "medicare",
    "elders",
}

def main() -> None:
    texts, labels = load_clean_dataset()
    baseline_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
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
    artifact_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=list(BLOG_ARTIFACTS)
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
    discourse_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words=list(DISCOURSE_ROLE_WORDS)
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

    clinical_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=list(CLINICAL_TOPIC_WORDS)
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
    baseline_results = cross_validate(
        baseline_pipeline,
        texts,
        labels,
        cv=cv,
        scoring=scoring,
    )

    artifact_results = cross_validate(
        artifact_pipeline,
        texts,
        labels,
        cv=cv,
        scoring=scoring,
    )

    discourse_results = cross_validate(
    discourse_pipeline,
    texts,
    labels,
    cv=cv,
    scoring=scoring,
)
    clinical_results = cross_validate(
        clinical_pipeline,
        texts,
        labels,
        cv=cv,
        scoring=scoring,
    )
    
    baseline_scores = baseline_results["test_macro_f1"]
    artifact_scores = artifact_results["test_macro_f1"]
    discourse_scores = discourse_results["test_macro_f1"]
    clinical_scores = clinical_results["test_macro_f1"]

    print("=== BASELINE ===")
    print("Fold:", np.round(baseline_scores, 4))
    print("Media:", round(baseline_scores.mean(), 4))
    print("Std:", round(baseline_scores.std(), 4))

    print()

    print("=== WITHOUT BLOG/WEB ARTIFACTS ===")
    print("Fold:", np.round(artifact_scores, 4))
    print("Media:", round(artifact_scores.mean(), 4))
    print("Std:", round(artifact_scores.std(), 4))

    print()

    difference = artifact_scores.mean() - baseline_scores.mean()

    print("Differenza Macro-F1:", round(difference, 4))
    print()
    print("=== WITHOUT DISCOURSE-ROLE CUES ===")
    print("Fold:", np.round(discourse_scores, 4))
    print("Media:", round(discourse_scores.mean(), 4))
    print("Std:", round(discourse_scores.std(), 4))

    discourse_difference = (
        discourse_scores.mean()
        - baseline_scores.mean()
    )

    print(
        "Differenza Macro-F1:",
        round(discourse_difference, 4),
    )

    print()
    print("=== WITHOUT CLINICAL TOPIC WORDS ===")
    print("Fold:", np.round(clinical_scores, 4))
    print("Media:", round(clinical_scores.mean(), 4))
    print("Std:", round(clinical_scores.std(), 4))

    clinical_difference = (
        clinical_scores.mean()
        - baseline_scores.mean()
    )

    print(
        "Differenza Macro-F1:",
        round(clinical_difference, 4),
    )

if __name__ == "__main__":
    main()
