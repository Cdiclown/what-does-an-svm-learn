import csv

import numpy as np


INPUT_FILE = "linguistic_error_analysis.csv"


LENGTH_BINS = [
    (0, 19, "<20"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 199, "100-199"),
    (200, 499, "200-499"),
    (500, float("inf"), "500+"),
]


def count_words(text: str) -> int:
    return len(
        text.split()
    )


def main() -> None:

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            row["true_label"] = int(
                row["true_label"]
            )

            row["predicted_label"] = int(
                row["predicted_label"]
            )

            row["correct"] = int(
                row["correct"]
            )

            row["num_words"] = count_words(
                row["text"]
            )

            rows.append(row)

    print(
        "Documenti:",
        len(rows),
    )

    print()
    print(
        "=== ERROR RATE BY DOCUMENT LENGTH ==="
    )

    for minimum, maximum, name in LENGTH_BINS:

        subset = [
            row
            for row in rows
            if (
                row["num_words"] >= minimum
                and row["num_words"] <= maximum
            )
        ]

        if not subset:
            continue

        num_documents = len(subset)

        num_errors = sum(
            row["correct"] == 0
            for row in subset
        )

        error_rate = (
            num_errors / num_documents
        )

        dementia_count = sum(
            row["true_label"] == 1
            for row in subset
        )

        control_count = sum(
            row["true_label"] == 0
            for row in subset
        )

        print()
        print(name)

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
            "Errori:",
            num_errors,
        )

        print(
            "Error rate:",
            round(
                error_rate,
                4,
            ),
        )

    # -----------------------------------------------
    # Accuracy per soglie minime
    # -----------------------------------------------

    print()
    print(
        "=== MINIMUM LENGTH THRESHOLDS ==="
    )

    for threshold in [
        20,
        50,
        100,
        200,
    ]:

        subset = [
            row
            for row in rows
            if row["num_words"] >= threshold
        ]

        num_documents = len(subset)

        num_correct = sum(
            row["correct"] == 1
            for row in subset
        )

        accuracy = (
            num_correct / num_documents
        )

        print()
        print(
            f">= {threshold} words"
        )

        print(
            "Documenti:",
            num_documents,
        )

        print(
            "Accuracy OOF:",
            round(
                accuracy,
                4,
            ),
        )


if __name__ == "__main__":
    main()