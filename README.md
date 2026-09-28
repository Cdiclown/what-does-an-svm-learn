# What Does an SVM Learn from Blogs Written by People with Alzheimer’s?

### Investigating lexical shortcuts, structural cues, and interpretable linguistic representations

This repository contains the code and experimental report developed for the
**Machine Learning for NLP** course at the **University of Trento (UniTN)**.

The project investigates a simple but important question:

> **Does a highly accurate text classifier actually learn linguistic patterns
> related to Alzheimer's disease, or does it mainly exploit dataset-specific
> shortcuts?**

Rather than focusing only on predictive performance, the project analyzes
**what information a linear SVM can exploit under different representations
of the same texts**.

Three representations are compared:

- standard **TF-IDF**, preserving the complete lexical content;
- a **delexicalized representation**, where content words are replaced by
  part-of-speech tags;
- a compact set of **22 interpretable linguistic features** inspired by
  psycholinguistic research on Alzheimer's-related language.

---

## Project Context

This project was developed as part of the **Machine Learning for NLP** course
at the **University of Trento**.

It builds on the text-classification practical work and experimental methodology
introduced during the course, extending the original classification task toward
a more detailed analysis of **model behavior, dataset shortcuts, linguistic
representations, and interpretability**.

Related course material:

[Machine Learning for NLP — Authorship Attribution Tutorial](https://github.com/ml-for-nlp/authorship-attribution)

The linked course repository provides a simple text-classification tutorial
based on authorship attribution and Naive Bayes. This project follows the
broader course approach of experimenting with text representations and
interpreting discriminative features, while investigating a different
classification problem using linear SVMs.

---

## Research Question

Text classifiers can achieve very high accuracy even when part of their
predictive power comes from properties that are only indirectly related to the
phenomenon of interest.

This is especially relevant for the dataset used in this project.

The two classes contain:

- blog posts associated with people with **Alzheimer's disease**;
- posts written by **relatives, caregivers, or unaffected controls**.

This creates several possible confounding factors.

For example, a classifier could distinguish the two groups using:

- differences in discourse role;
- patient vs. caregiver perspective;
- individual author style;
- medical terminology;
- blog-specific artifacts;
- recurring topics;

rather than linguistic manifestations of cognitive decline.

The main research question is therefore:

> **Does the classifier detect Alzheimer-related linguistic patterns, or does
> it mainly exploit dataset-specific shortcuts?**

Because author identities are not available in the dataset, this experiment
cannot fully separate genuine cognitive effects from author-specific or
corpus-specific effects.

The goal is therefore more precisely to investigate **how much classification
performance depends on lexical content and how much discriminative information
remains in structural and interpretable linguistic properties**.

---

## Dataset

After removing seven empty documents, the dataset contains:

- **3,644 blog posts**
- **2,274 Alzheimer-class documents**
- **1,370 control-class documents**

The classes are therefore moderately imbalanced.

All models are evaluated using:

- **5-fold stratified cross-validation**
- fixed random seed (`42`)
- balanced SVM class weights

The main evaluation metric is **Macro-F1**, together with balanced accuracy.

A limitation of the dataset is that the experiments use a document-level split.

Ideally, evaluation should be performed using an **author-level split**, so
that posts from the same author cannot appear in both training and testing.
However, author identifiers are not available in the provided dataset.

---

# Experimental Design

The same classifier — a **linear Support Vector Machine (SVM)** — is used
throughout the project.

Keeping the classifier fixed makes it possible to study the effect of changing
the **representation of the text** rather than changing the learning algorithm.

Three representations are compared.

---

## 1. TF-IDF Baseline

The first representation uses standard **document-level TF-IDF**.

TF-IDF is fitted only on the training portion of each cross-validation fold to
avoid information leakage.

The resulting representation contains approximately **26,000 lexical
features**.

The linear SVM achieves:

**Macro-F1: 0.9808**

Such high performance motivates the rest of the analysis.

Rather than immediately interpreting this score as evidence that the classifier
recognizes linguistic effects of Alzheimer's disease, the project investigates
whether the model is exploiting easier corpus-specific signals.

---

## 2. Lexical Ablation Experiments

The coefficients of the TF-IDF SVM are inspected to identify highly
discriminative lexical features.

Several possible shortcut categories emerge:

- **identity and web-domain artifacts**;
- **discourse-role cues**;
- **clinical terminology**.

Three ablation experiments therefore remove terms belonging to these categories
before retraining the model.

Interestingly, performance decreases only marginally.

This suggests that the predictive signal is not concentrated in a small number
of obvious shortcut words, but is instead **distributed across a much larger
portion of the vocabulary**.

---

## 3. Delexicalized Representation

To test whether lexical semantics are necessary for classification, a second
representation removes most explicit lexical content.

Using **spaCy POS tagging**:

- content words are replaced by their part-of-speech tags;
- function words are retained;
- pronouns are retained;
- punctuation is retained.

The resulting sequences are again represented using TF-IDF and classified with
the same linear SVM.

The model achieves:

**Macro-F1: 0.9529**

This is only around three Macro-F1 points below the full lexical model.

The result suggests that specific content words are **not necessary for
near-perfect separation of the two classes**.

Structural characteristics such as grammatical patterns, function words,
pronouns, punctuation, and POS sequences already contain a very strong
classification signal.

However, this should not automatically be interpreted as evidence of cognitive
decline.

The remaining signal could also reflect:

- individual writing style;
- author identity;
- discourse role;
- differences between blogs.

---

## 4. Interpretable Linguistic Model

The third representation deliberately compresses every document into only
**22 linguistic features**.

The features were selected a priori from several linguistic dimensions and
from previous research on language associated with cognitive decline.

### Lexical Features

- average word length
- Moving-Average Type-Token Ratio (**MATTR**)
- repetition rate
- vague-word rate

### Syntactic Features

- average sentence length
- sentence-length variability
- average dependency depth
- subordination rate
- clause-to-sentence ratio

### POS and Reference Features

- pronoun rate
- noun/pronoun ratio
- noun rate
- verb rate
- adjective rate
- adverb rate
- function-word rate

### Discourse Features

- lexical overlap between adjacent sentences
- connective rate
- hedge rate

### Tense and Modal Features

- modal-verb rate
- past-tense rate
- present-tense rate

Each feature is standardized using statistics computed exclusively from the
training fold.

The same linear SVM and cross-validation protocol are then applied.

The model achieves:

**Macro-F1: 0.8225**

Despite compressing each document from thousands of lexical dimensions to just
22 interpretable values, the model retains substantial discriminative power.

## Repository Contents

This repository contains:

- data preprocessing code;
- TF-IDF + linear SVM experiments;
- lexical ablation experiments;
- POS-based delexicalization;
- linguistic feature extraction;
- feature-group experiments;
- cross-validation pipelines;
- coefficient analysis;
- feature correlation analysis;
- permutation importance;
- out-of-fold error analysis;
- figures and experimental results;
- the complete experimental report.

---

## Technologies

The project uses:

- **Python**
- **scikit-learn**
- **spaCy**
- **TF-IDF**
- **Support Vector Machines**
- **Part-of-Speech tagging**
- **cross-validation**
- **permutation feature importance**
- **statistical and linguistic feature analysis**

---

## Course

**Machine Learning for NLP**  
University of Trento

Related course repository:

https://github.com/ml-for-nlp/authorship-attribution

---

## Author

**Chiara Tosadori**

University of Trento


---
## License

### Code

Unless otherwise stated, all **source code** in this repository is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code, subject to the terms of the MIT License.

### Reports, Papers, and Written Content

Unless otherwise stated, all **reports, papers, reviews, documentation, and other original written content** in this repository are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Under CC BY 4.0, you are free to:

* **Share** — copy and redistribute the material in any medium or format.
* **Adapt** — remix, transform, and build upon the material, including for commercial purposes.

The following conditions apply:

* **Attribution** — you must give appropriate credit to the author, provide a link to the license, and indicate if changes were made.
* You may not imply that the author endorses you or your use of the material.

For the full license terms, see the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Unless otherwise stated, © 2026 [Your Name]. All rights reserved for materials not covered by the licenses above.
