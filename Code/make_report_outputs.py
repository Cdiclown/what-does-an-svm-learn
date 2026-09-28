from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("report_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. MAIN REPRESENTATION COMPARISON
# ============================================================

main_results = pd.DataFrame({
    "Representation": [
        "TF-IDF",
        "Delexicalized",
        "Linguistic features",
    ],
    "Macro_F1": [
        0.9808,
        0.9529,
        0.8225,
    ],
    "Std": [
        0.0070,
        0.0041,
        0.0119,
    ],
    "Balanced_Accuracy": [
        0.9821,
        0.9588,
        0.8325,
    ],
})


main_results.to_csv(
    OUTPUT_DIR / "table_main_results.csv",
    index=False,
)


plt.figure(figsize=(7, 5))

plt.bar(
    main_results["Representation"],
    main_results["Macro_F1"],
    yerr=main_results["Std"],
    capsize=5,
)

plt.ylabel("Macro-F1")
plt.xlabel("Input representation")
plt.ylim(0.75, 1.0)

plt.title(
    "Classification performance across representations"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_1_representation_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# 2. FEATURE GROUP ANALYSIS
# ============================================================

feature_groups = pd.DataFrame({
    "Group": [
        "All 22",
        "POS / reference",
        "Lexical",
        "Syntactic",
        "Discourse",
        "Tense / modal",
    ],
    "N_features": [
        22,
        7,
        4,
        5,
        3,
        3,
    ],
    "Macro_F1": [
        0.8225,
        0.7299,
        0.6960,
        0.6811,
        0.6299,
        0.6042,
    ],
    "Std": [
        0.0119,
        0.0121,
        0.0074,
        0.0143,
        0.0212,
        0.0167,
    ],
    "Balanced_Accuracy": [
        0.8325,
        0.7351,
        0.6992,
        0.6859,
        0.6389,
        0.6100,
    ],
})


feature_groups.to_csv(
    OUTPUT_DIR / "table_feature_groups.csv",
    index=False,
)


plt.figure(figsize=(8, 5))

plt.bar(
    feature_groups["Group"],
    feature_groups["Macro_F1"],
    yerr=feature_groups["Std"],
    capsize=4,
)

plt.ylabel("Macro-F1")
plt.xlabel("Feature group")
plt.ylim(0.55, 0.86)

plt.title(
    "Performance of linguistic feature groups"
)

plt.xticks(
    rotation=25,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_2_feature_groups.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. PERMUTATION IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": [
        "function_word_rate",
        "noun_rate",
        "past_verb_rate",
        "mattr",
        "adverb_rate",
        "avg_dependency_depth",
        "connective_rate",
        "pronoun_rate",
        "adjacent_sentence_overlap",
        "repetition_rate",
        "subordination_rate",
        "avg_word_length",
    ],
    "Mean_drop_F1": [
        0.1078,
        0.0777,
        0.0632,
        0.0559,
        0.0379,
        0.0290,
        0.0258,
        0.0249,
        0.0206,
        0.0159,
        0.0139,
        0.0115,
    ],
    "Std": [
        0.0314,
        0.0195,
        0.0179,
        0.0117,
        0.0108,
        0.0052,
        0.0071,
        0.0041,
        0.0030,
        0.0039,
        0.0058,
        0.0039,
    ],
})


importance.to_csv(
    OUTPUT_DIR / "table_permutation_importance.csv",
    index=False,
)


importance_plot = importance.sort_values(
    "Mean_drop_F1",
    ascending=True,
)


plt.figure(figsize=(8, 6))

plt.barh(
    importance_plot["Feature"],
    importance_plot["Mean_drop_F1"],
    xerr=importance_plot["Std"],
    capsize=3,
)

plt.xlabel(
    "Decrease in Macro-F1 after permutation"
)

plt.ylabel("Feature")

plt.title(
    "Permutation importance of linguistic features"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_3_permutation_importance.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. DOCUMENT LENGTH ROBUSTNESS
# ============================================================

length_results = pd.DataFrame({
    "Minimum_words": [
        0,
        20,
        50,
        100,
        200,
    ],
    "Documents": [
        3644,
        3562,
        3363,
        3079,
        2458,
    ],
    "Macro_F1": [
        0.8225,
        0.8231,
        0.8340,
        0.8402,
        0.8446,
    ],
    "Std": [
        0.0119,
        0.0200,
        0.0116,
        0.0159,
        0.0126,
    ],
    "Balanced_Accuracy": [
        0.8325,
        0.8322,
        0.8407,
        0.8474,
        0.8490,
    ],
})


length_results.to_csv(
    OUTPUT_DIR / "table_length_robustness.csv",
    index=False,
)


plt.figure(figsize=(7, 5))

plt.errorbar(
    length_results["Minimum_words"],
    length_results["Macro_F1"],
    yerr=length_results["Std"],
    marker="o",
    capsize=4,
)

plt.xlabel(
    "Minimum document length (words)"
)

plt.ylabel("Macro-F1")

plt.title(
    "Robustness to minimum document length"
)

plt.ylim(0.79, 0.87)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_4_length_robustness.png",
    dpi=300,
)

plt.close()


# ============================================================
# 5. ERROR AGREEMENT TABLE
# ============================================================

error_agreement = pd.DataFrame({
    "Pattern": [
        "All correct",
        "TF-IDF + Delex only",
        "TF-IDF only",
        "Delex + Linguistic only",
        "Delex only",
        "Linguistic only",
        "TF-IDF + Linguistic only",
        "All wrong",
    ],
    "Documents": [
        2956,
        499,
        81,
        18,
        8,
        5,
        42,
        35,
    ],
})


error_agreement.to_csv(
    OUTPUT_DIR / "table_error_agreement.csv",
    index=False,
)


# ============================================================
# 6. DATASET SUMMARY TABLE
# ============================================================

dataset_summary = pd.DataFrame({
    "Statistic": [
        "Usable documents",
        "Dementia-class documents",
        "Control-class documents",
        "Dementia mean length",
        "Control mean length",
        "Dementia median length",
        "Control median length",
    ],
    "Value": [
        3644,
        2274,
        1370,
        277.54,
        517.37,
        246.0,
        526.5,
    ],
})


dataset_summary.to_csv(
    OUTPUT_DIR / "table_dataset_summary.csv",
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print()
print("Report outputs created in:")
print(OUTPUT_DIR.resolve())

print()
print("Figures:")
print("  figure_1_representation_comparison.png")
print("  figure_2_feature_groups.png")
print("  figure_3_permutation_importance.png")
print("  figure_4_length_robustness.png")

print()
print("Tables:")
print("  table_main_results.csv")
print("  table_feature_groups.csv")
print("  table_permutation_importance.csv")
print("  table_length_robustness.csv")
print("  table_error_agreement.csv")
print("  table_dataset_summary.csv")