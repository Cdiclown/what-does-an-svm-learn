import numpy as np
import spacy

from baseline import load_clean_dataset


# --------------------------------------------------
# MODELLO NLP
# --------------------------------------------------

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner"],
)


# --------------------------------------------------
# COSTANTI
# --------------------------------------------------

VAGUE_WORDS = {
    "thing",
    "things",
    "something",
    "stuff",
}


SUBORDINATE_DEPENDENCIES = {
    "advcl",
    "ccomp",
    "xcomp",
    "acl",
    "relcl",
    "csubj",
}


CONTENT_POS = {
    "NOUN",
    "PROPN",
    "VERB",
    "ADJ",
    "ADV",
}


# POS tipicamente associati a function words.
FUNCTION_POS = {
    "ADP",
    "AUX",
    "CCONJ",
    "DET",
    "PART",
    "PRON",
    "SCONJ",
}


CONNECTIVES = {
    "although",
    "because",
    "but",
    "consequently",
    "finally",
    "furthermore",
    "hence",
    "however",
    "instead",
    "meanwhile",
    "moreover",
    "nevertheless",
    "next",
    "otherwise",
    "since",
    "so",
    "then",
    "therefore",
    "though",
    "thus",
    "while",
    "yet",
}


HEDGE_SINGLE_WORDS = {
    "apparently",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "seem",
    "seems",
    "seemed",
    "somehow",
}


HEDGE_PHRASES = {
    ("i", "think"),
    ("i", "guess"),
    ("i", "suppose"),
    ("i", "believe"),
    ("you", "know"),
    ("kind", "of"),
    ("sort", "of"),
}


MODAL_LEMMAS = {
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}


MATTR_WINDOW = 50
REPETITION_WINDOW = 50


# --------------------------------------------------
# DEPENDENCY DEPTH
# --------------------------------------------------

def get_token_depth(token) -> int:

    depth = 0
    current = token

    while current.head != current:
        depth += 1
        current = current.head

    return depth


# --------------------------------------------------
# MATTR
# --------------------------------------------------

def calculate_mattr(
    words,
    window_size: int = MATTR_WINDOW,
) -> float:

    lemmas = [
        token.lemma_.lower()
        for token in words
    ]

    num_tokens = len(lemmas)

    if num_tokens == 0:
        return 0.0

    if num_tokens <= window_size:

        return (
            len(set(lemmas))
            / num_tokens
        )

    values = []

    for start in range(
        num_tokens - window_size + 1
    ):

        window = lemmas[
            start:start + window_size
        ]

        ttr = (
            len(set(window))
            / window_size
        )

        values.append(ttr)

    return float(
        np.mean(values)
    )


# --------------------------------------------------
# REPETITION RATE
# --------------------------------------------------

def calculate_repetition_rate(
    words,
    window_size: int = REPETITION_WINDOW,
) -> float:

    content_lemmas = [
        token.lemma_.lower()
        for token in words
        if token.pos_ in CONTENT_POS
    ]

    num_tokens = len(
        content_lemmas
    )

    if num_tokens == 0:
        return 0.0

    if num_tokens <= window_size:

        unique = len(
            set(content_lemmas)
        )

        return (
            num_tokens - unique
        ) / num_tokens

    values = []

    for start in range(
        num_tokens - window_size + 1
    ):

        window = content_lemmas[
            start:start + window_size
        ]

        unique = len(
            set(window)
        )

        repetition = (
            len(window) - unique
        ) / len(window)

        values.append(
            repetition
        )

    return float(
        np.mean(values)
    )


# --------------------------------------------------
# ADJACENT SENTENCE OVERLAP
# --------------------------------------------------

def calculate_adjacent_sentence_overlap(
    sentences,
) -> float:

    sentence_sets = []

    for sentence in sentences:

        content_lemmas = {
            token.lemma_.lower()
            for token in sentence
            if (
                token.is_alpha
                and token.pos_ in CONTENT_POS
            )
        }

        sentence_sets.append(
            content_lemmas
        )

    values = []

    for index in range(
        len(sentence_sets) - 1
    ):

        first = sentence_sets[index]
        second = sentence_sets[index + 1]

        if not first or not second:
            continue

        intersection = (
            first & second
        )

        union = (
            first | second
        )

        values.append(
            len(intersection)
            / len(union)
        )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# --------------------------------------------------
# CONNECTIVE RATE
# --------------------------------------------------

def calculate_connective_rate(
    doc,
    num_sentences: int,
) -> float:

    if num_sentences == 0:
        return 0.0

    count = sum(
        token.lower_ in CONNECTIVES
        for token in doc
        if token.is_alpha
    )

    return (
        count / num_sentences
    )


# --------------------------------------------------
# HEDGE RATE
# --------------------------------------------------

def calculate_hedge_rate(
    doc,
    num_words: int,
) -> float:

    if num_words == 0:
        return 0.0

    tokens = [
        token.lower_
        for token in doc
        if token.is_alpha
    ]

    hedge_count = 0

    # Hedge singoli
    hedge_count += sum(
        token in HEDGE_SINGLE_WORDS
        for token in tokens
    )

    # Hedge multiword
    for phrase in HEDGE_PHRASES:

        phrase_length = len(phrase)

        for start in range(
            len(tokens)
            - phrase_length
            + 1
        ):

            candidate = tuple(
                tokens[
                    start:
                    start + phrase_length
                ]
            )

            if candidate == phrase:
                hedge_count += 1

    return (
        hedge_count / num_words
    )


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(doc) -> dict:

    words = [
        token
        for token in doc
        if token.is_alpha
    ]

    num_words = len(words)

    if num_words == 0:
        return {
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "avg_dependency_depth": 0.0,
            "subordination_rate": 0.0,
            "clause_to_sentence_ratio": 0.0,
            "avg_word_length": 0.0,
            "mattr": 0.0,
            "repetition_rate": 0.0,
            "pronoun_rate": 0.0,
            "noun_pronoun_ratio": 0.0,
            "noun_rate": 0.0,
            "verb_rate": 0.0,
            "adjective_rate": 0.0,
            "adverb_rate": 0.0,
            "function_word_rate": 0.0,
            "vague_word_rate": 0.0,
            "adjacent_sentence_overlap": 0.0,
            "connective_rate": 0.0,
            "hedge_rate": 0.0,
            "modal_verb_rate": 0.0,
            "past_verb_rate": 0.0,
            "present_verb_rate": 0.0,
        }

    # --------------------------------------------------
    # FRASI
    # --------------------------------------------------

    sentences = list(
        doc.sents
    )

    sentence_lengths = []
    sentence_depths = []

    for sentence in sentences:

        sentence_words = [
            token
            for token in sentence
            if token.is_alpha
        ]

        if not sentence_words:
            continue

        sentence_lengths.append(
            len(sentence_words)
        )

        max_depth = max(
            get_token_depth(token)
            for token in sentence_words
        )

        sentence_depths.append(
            max_depth
        )

    num_sentences = len(
        sentence_lengths
    )

    # --------------------------------------------------
    # 1. AVG SENTENCE LENGTH
    # --------------------------------------------------

    avg_sentence_length = (
        np.mean(sentence_lengths)
        if sentence_lengths
        else 0.0
    )

    # --------------------------------------------------
    # 2. SENTENCE LENGTH STD
    # --------------------------------------------------

    sentence_length_std = (
        float(np.std(sentence_lengths))
        if sentence_lengths
        else 0.0
    )

    # --------------------------------------------------
    # 3. AVG DEPENDENCY DEPTH
    # --------------------------------------------------

    avg_dependency_depth = (
        np.mean(sentence_depths)
        if sentence_depths
        else 0.0
    )

    # --------------------------------------------------
    # 4. SUBORDINATION RATE
    # --------------------------------------------------

    subordinate_count = sum(
        token.dep_
        in SUBORDINATE_DEPENDENCIES
        for token in doc
    )

    subordination_rate = (
        subordinate_count
        / num_sentences
        if num_sentences > 0
        else 0.0
    )

    # --------------------------------------------------
    # 5. CLAUSE-TO-SENTENCE RATIO
    # --------------------------------------------------

    # Consideriamo un verbo finito come
    # proxy di una clausola.
    finite_verb_count = sum(
        (
            token.pos_ in {"VERB", "AUX"}
            and "Fin"
            in token.morph.get(
                "VerbForm"
            )
        )
        for token in doc
    )

    clause_to_sentence_ratio = (
        finite_verb_count
        / num_sentences
        if num_sentences > 0
        else 0.0
    )

    # --------------------------------------------------
    # 6. AVG WORD LENGTH
    # --------------------------------------------------

    avg_word_length = (
        sum(
            len(token.text)
            for token in words
        )
        / num_words
    )

    # --------------------------------------------------
    # 7. MATTR
    # --------------------------------------------------

    mattr = calculate_mattr(
        words
    )

    # --------------------------------------------------
    # 8. REPETITION RATE
    # --------------------------------------------------

    repetition_rate = (
        calculate_repetition_rate(
            words
        )
    )

    # --------------------------------------------------
    # POS COUNTS
    # --------------------------------------------------

    pronoun_count = sum(
        token.pos_ == "PRON"
        for token in words
    )

    noun_count = sum(
        token.pos_
        in {"NOUN", "PROPN"}
        for token in words
    )

    verb_count = sum(
        token.pos_ == "VERB"
        for token in words
    )

    adjective_count = sum(
        token.pos_ == "ADJ"
        for token in words
    )

    adverb_count = sum(
        token.pos_ == "ADV"
        for token in words
    )

    # --------------------------------------------------
    # 9. PRONOUN RATE
    # --------------------------------------------------

    pronoun_rate = (
        pronoun_count
        / num_words
    )

    # --------------------------------------------------
    # 10. NOUN / PRONOUN RATIO
    # --------------------------------------------------

    noun_pronoun_ratio = (
        noun_count
        / pronoun_count
        if pronoun_count > 0
        else 0.0
    )

    # --------------------------------------------------
    # 11. NOUN RATE
    # --------------------------------------------------

    noun_rate = (
        noun_count
        / num_words
    )

    # --------------------------------------------------
    # 12. VERB RATE
    # --------------------------------------------------

    verb_rate = (
        verb_count
        / num_words
    )

    # --------------------------------------------------
    # 13. ADJECTIVE RATE
    # --------------------------------------------------

    adjective_rate = (
        adjective_count
        / num_words
    )

    # --------------------------------------------------
    # 14. ADVERB RATE
    # --------------------------------------------------

    adverb_rate = (
        adverb_count
        / num_words
    )

    # --------------------------------------------------
    # 15. FUNCTION WORD RATE
    # --------------------------------------------------

    function_word_count = sum(
        token.pos_ in FUNCTION_POS
        for token in words
    )

    function_word_rate = (
        function_word_count
        / num_words
    )

    # --------------------------------------------------
    # 16. VAGUE-WORD RATE
    # --------------------------------------------------

    vague_count = sum(
        token.lower_
        in VAGUE_WORDS
        for token in words
    )

    vague_word_rate = (
        vague_count
        / num_words
    )

    # --------------------------------------------------
    # 17. ADJACENT SENTENCE OVERLAP
    # --------------------------------------------------

    adjacent_sentence_overlap = (
        calculate_adjacent_sentence_overlap(
            sentences
        )
    )

    # --------------------------------------------------
    # 18. CONNECTIVE RATE
    # --------------------------------------------------

    connective_rate = (
        calculate_connective_rate(
            doc,
            num_sentences,
        )
    )

    # --------------------------------------------------
    # 19. HEDGE RATE
    # --------------------------------------------------

    hedge_rate = (
        calculate_hedge_rate(
            doc,
            num_words,
        )
    )

    # --------------------------------------------------
    # 20. MODAL VERB RATE
    # --------------------------------------------------

    modal_count = sum(
        token.lemma_.lower()
        in MODAL_LEMMAS
        for token in words
        if token.pos_ == "AUX"
    )

    modal_verb_rate = (
        modal_count
        / num_words
    )

    # --------------------------------------------------
    # 21-22. VERB TENSE DISTRIBUTION
    # --------------------------------------------------

    # Consideriamo VERB e AUX per cui
    # spaCy riconosce esplicitamente
    # Past oppure Pres.

    past_count = 0
    present_count = 0

    tense_marked_verb_count = 0

    for token in words:

        if token.pos_ not in {
            "VERB",
            "AUX",
        }:
            continue

        tense = token.morph.get(
            "Tense"
        )

        if "Past" in tense:
            past_count += 1
            tense_marked_verb_count += 1

        elif "Pres" in tense:
            present_count += 1
            tense_marked_verb_count += 1

    if tense_marked_verb_count > 0:

        past_verb_rate = (
            past_count
            / tense_marked_verb_count
        )

        present_verb_rate = (
            present_count
            / tense_marked_verb_count
        )

    else:

        past_verb_rate = 0.0
        present_verb_rate = 0.0

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    return {
        "avg_sentence_length":
            float(avg_sentence_length),

        "sentence_length_std":
            sentence_length_std,

        "avg_dependency_depth":
            float(avg_dependency_depth),

        "subordination_rate":
            subordination_rate,

        "clause_to_sentence_ratio":
            clause_to_sentence_ratio,

        "avg_word_length":
            avg_word_length,

        "mattr":
            mattr,

        "repetition_rate":
            repetition_rate,

        "pronoun_rate":
            pronoun_rate,

        "noun_pronoun_ratio":
            noun_pronoun_ratio,

        "noun_rate":
            noun_rate,

        "verb_rate":
            verb_rate,

        "adjective_rate":
            adjective_rate,

        "adverb_rate":
            adverb_rate,

        "function_word_rate":
            function_word_rate,

        "vague_word_rate":
            vague_word_rate,

        "adjacent_sentence_overlap":
            adjacent_sentence_overlap,

        "connective_rate":
            connective_rate,

        "hedge_rate":
            hedge_rate,

        "modal_verb_rate":
            modal_verb_rate,

        "past_verb_rate":
            past_verb_rate,

        "present_verb_rate":
            present_verb_rate,
    }


# --------------------------------------------------
# ESTRAZIONE TUTTO DATASET
# --------------------------------------------------

def extract_all_features(
    texts: list[str]
) -> list[dict]:

    all_features = []

    for doc in nlp.pipe(
        texts,
        batch_size=32,
    ):

        all_features.append(
            extract_features(doc)
        )

    return all_features


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    texts, labels = (
        load_clean_dataset()
    )

    print(
        "Estrazione delle feature..."
    )

    all_features = (
        extract_all_features(
            texts
        )
    )

    print(
        "Documenti analizzati:",
        len(all_features),
    )

    feature_names = [
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

    print()
    print(
        "=== CONFRONTO TRA LE CLASSI ==="
    )

    for feature_name in feature_names:

        dementia_values = [
            features[feature_name]
            for features, label
            in zip(
                all_features,
                labels,
            )
            if label == 1
        ]

        control_values = [
            features[feature_name]
            for features, label
            in zip(
                all_features,
                labels,
            )
            if label == 0
        ]

        print()
        print(feature_name)

        print(
            "Dementia - mean:",
            round(
                np.mean(
                    dementia_values
                ),
                4,
            ),
        )

        print(
            "Dementia - median:",
            round(
                np.median(
                    dementia_values
                ),
                4,
            ),
        )

        print(
            "Control  - mean:",
            round(
                np.mean(
                    control_values
                ),
                4,
            ),
        )

        print(
            "Control  - median:",
            round(
                np.median(
                    control_values
                ),
                4,
            ),
        )


if __name__ == "__main__":
    main()