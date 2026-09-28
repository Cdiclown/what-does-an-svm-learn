import spacy


nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner"],
)


CONTENT_POS = {
    "NOUN",
    "PROPN",
    "VERB",
    "ADJ",
    "ADV",
    "NUM",
}


def delexicalize(text: str) -> str:
    doc = nlp(text)

    new_tokens = []

    for token in doc:
        if token.is_space:
            continue

        if token.pos_ in CONTENT_POS:
            new_tokens.append(token.pos_)
        else:
            new_tokens.append(token.text.lower())

    return " ".join(new_tokens)

def delexicalize_documents(texts: list[str]) -> list[str]:
    delexicalized_texts = []

    for doc in nlp.pipe(texts, batch_size=32):

        new_tokens = []

        for token in doc:
            if token.is_space:
                continue

            if token.pos_ in CONTENT_POS:
                new_tokens.append(token.pos_)
            else:
                new_tokens.append(token.text.lower())

        delexicalized_texts.append(
            " ".join(new_tokens)
        )

    return delexicalized_texts

if __name__ == "__main__":

    examples = [
        "My mother forgot the appointment yesterday.",
        "I went to Atlanta with my wife.",
        "I couldn't remember what that thing was, so I said something.",
    ]

    for text in examples:
        print()
        print("ORIGINAL:")
        print(text)

        print("DELEXICALIZED:")
        print(delexicalize(text))
