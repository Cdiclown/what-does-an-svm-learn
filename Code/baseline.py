from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from inspect_dataset import parse_linear_file
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import numpy as np
from sklearn.pipeline import Pipeline

 
def load_clean_dataset() -> tuple[list[str], list[int]]:
    class1_documents = parse_linear_file(
        file_path="data/class1/linear.txt",
        label=1,
        class_name="class1",
    )

    class2_documents = parse_linear_file(
        file_path="data/class2/linear.txt",
        label=0,
        class_name="class2",
    )

    all_documents = class1_documents + class2_documents

    clean_documents = [
        document
        for document in all_documents
        if document["text"].strip()
    ]

    texts = [
        document["text"]
        for document in clean_documents
    ]

    labels = [
        document["label"]
        for document in clean_documents
    ]

    return texts, labels

def main() -> None:
    texts, labels = load_clean_dataset()

    x_train, x_test, y_train, y_test = train_test_split( #train_test_split divide il dataset in training e test
        texts,
        labels,
        test_size=0.20,
        random_state=42, #blocca il random per avere sempre lo stesso split
        stratify=labels, #Serve a mantenere la stessa proporzione delle classi.


    )


    pipeline = Pipeline([
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
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    tfidf = pipeline.named_steps["tfidf"]
    svm = pipeline.named_steps["svm"]
        
    feature_names = tfidf.get_feature_names_out()

    coefficients = svm.coef_[0]

    top_positive_indices = np.argsort(coefficients)[-50:][::-1]
    top_negative_indices = np.argsort(coefficients)[:50]

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

    cv_results = cross_validate(
    pipeline,
    texts,
    labels,
    cv=cv,
    scoring=scoring,
)


    print()
    print("=== TOP FEATURE DEMENTIA ===")

    for index in top_positive_indices:
     print(
        feature_names[index],
        round(coefficients[index], 4),
    )

    print()
    print("=== TOP FEATURE CONTROL ===")

    for index in top_negative_indices:
        print(
            feature_names[index],
            round(coefficients[index], 4),
        )
    print("=== 5-FOLD CROSS-VALIDATION ===")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        scores = cv_results[f"test_{metric}"]

        print()
        print(metric)
        print("Fold:", np.round(scores, 4))
        print("Media:", round(scores.mean(), 4))
        print("Deviazione standard:", round(scores.std(), 4))

if __name__ == "__main__":
    main()