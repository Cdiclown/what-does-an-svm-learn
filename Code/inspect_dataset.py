from collections import Counter
from pathlib import Path
from statistics import mean, median


def parse_linear_file(
    file_path: str,
    label: int,
    class_name: str,
) -> list[dict]:
    """
    Legge un file linear.txt e restituisce una lista di documenti.

    Ogni documento contiene:
    - doc_id: identificatore univoco;
    - date: data del post;
    - text: testo completo del post;
    - label: etichetta numerica;
    - class_name: nome della classe.
    """

    documents = []

    inside_document = False
    current_date = None
    current_lines = []
    document_index = 0

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open(
        mode="r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")

            # Inizio di un nuovo documento
            if line.startswith("<doc date="):
                if inside_document:
                    raise ValueError(
                        f"Nuovo <doc> trovato prima della chiusura "
                        f"del documento precedente, riga {line_number}"
                    )

                prefix = "<doc date="

                if not line.endswith(">"):
                    raise ValueError(
                        f"Tag <doc> non chiuso correttamente "
                        f"alla riga {line_number}"
                    )

                current_date = line[len(prefix):-1]
                current_lines = []
                inside_document = True

            # Fine del documento corrente
            elif line.strip() == "</doc>":
                if not inside_document:
                    raise ValueError(
                        f"Tag </doc> senza un tag <doc> corrispondente "
                        f"alla riga {line_number}"
                    )

                text = "\n".join(current_lines).strip()
                doc_id = f"{class_name}_{document_index:05d}"

                documents.append(
                    {
                        "doc_id": doc_id,
                        "date": current_date,
                        "text": text,
                        "label": label,
                        "class_name": class_name,
                    }
                )

                document_index += 1
                current_date = None
                current_lines = []
                inside_document = False

            # Riga appartenente al testo del documento
            elif inside_document:
                current_lines.append(line)

            # Testo non vuoto trovato fuori dai tag
            elif line.strip():
                raise ValueError(
                    f"Testo fuori da un documento alla riga {line_number}"
                )

    if inside_document:
        raise ValueError(
            f"Il file {file_path} è terminato prima del tag </doc>"
        )

    return documents


def main() -> None:
    # Caricamento dei due file
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

    print("=== DIMENSIONE DEL DATASET ===")
    print("Documenti class1:", len(class1_documents))
    print("Documenti class2:", len(class2_documents))
    print("Documenti totali:", len(all_documents))

    # Individuazione dei documenti vuoti
    empty_documents = [
        document
        for document in all_documents
        if not document["text"].strip()
    ]

    print()
    print("=== CONTROLLO DOCUMENTI VUOTI ===")
    print("Documenti vuoti:", len(empty_documents))

    if empty_documents:
        print("Documenti vuoti trovati:")

        for document in empty_documents:
            print(
                f"- {document['doc_id']} | "
                f"{document['class_name']} | "
                f"{document['date']}"
            )

    # Dataset pulito: rimozione dei post vuoti
    clean_documents = [
        document
        for document in all_documents
        if document["text"].strip()
    ]

    print()
    print("=== PULIZIA DEL DATASET ===")
    print("Documenti prima della pulizia:", len(all_documents))
    print("Documenti dopo la pulizia:", len(clean_documents))
    print(
        "Documenti rimossi:",
        len(all_documents) - len(clean_documents),
    )

    # Controllo degli identificatori
    document_ids = [
        document["doc_id"]
        for document in all_documents
    ]

    unique_document_ids = set(document_ids)

    print()
    print("=== CONTROLLO IDENTIFICATORI ===")
    print("ID totali:", len(document_ids))
    print("ID univoci:", len(unique_document_ids))
    print(
        "ID duplicati:",
        len(document_ids) - len(unique_document_ids),
    )

    # Controllo delle date duplicate
    date_counts = Counter(
        document["date"]
        for document in all_documents
    )

    duplicate_dates = {
        date: count
        for date, count in date_counts.items()
        if count > 1
    }

    posts_with_shared_date = sum(
        count
        for count in duplicate_dates.values()
    )

    print()
    print("=== CONTROLLO DATE ===")
    print("Date distinte:", len(date_counts))
    print(
        "Date condivise da più post:",
        len(duplicate_dates),
    )
    print(
        "Post che condividono la data con almeno un altro post:",
        posts_with_shared_date,
    )

    print()
    print("Prime 10 date duplicate:")

    for date, count in list(duplicate_dates.items())[:10]:
        print(f"- {date}: {count} post")

    # Conteggio delle classi dopo la pulizia
    clean_class1 = [
        document
        for document in clean_documents
        if document["label"] == 1
    ]

    clean_class2 = [
        document
        for document in clean_documents
        if document["label"] == 0
    ]

    print()
    print("=== DISTRIBUZIONE DELLE CLASSI ===")
    print("Class1 pulita:", len(clean_class1))
    print("Class2 pulita:", len(clean_class2))

    class1_percentage = (
        len(clean_class1) / len(clean_documents) * 100
    )

    class2_percentage = (
        len(clean_class2) / len(clean_documents) * 100
    )

    print(
        "Percentuale class1:",
        round(class1_percentage, 2),
        "%",
    )
    print(
        "Percentuale class2:",
        round(class2_percentage, 2),
        "%",
    )

    # Calcolo della lunghezza di ogni documento
    for document in clean_documents:
        document["num_characters"] = len(document["text"])
        document["num_words"] = len(document["text"].split())

    class1_word_lengths = [
        document["num_words"]
        for document in clean_documents
        if document["label"] == 1
    ]

    class2_word_lengths = [
        document["num_words"]
        for document in clean_documents
        if document["label"] == 0
    ]

    print()
    print("=== LUNGHEZZA DEI POST ===")

    print("Class1:")
    print(
        "- media parole:",
        round(mean(class1_word_lengths), 2),
    )
    print(
        "- mediana parole:",
        median(class1_word_lengths),
    )
    print(
        "- minimo parole:",
        min(class1_word_lengths),
    )
    print(
        "- massimo parole:",
        max(class1_word_lengths),
    )

    print()
    print("Class2:")
    print(
        "- media parole:",
        round(mean(class2_word_lengths), 2),
    )
    print(
        "- mediana parole:",
        median(class2_word_lengths),
    )
    print(
        "- minimo parole:",
        min(class2_word_lengths),
    )
    print(
        "- massimo parole:",
        max(class2_word_lengths),
    )

    # Esempio di un documento per classe
    print()
    print("=== ESEMPIO CLASS1 ===")

    first_class1 = clean_class1[0]

    print("ID:", first_class1["doc_id"])
    print("Label:", first_class1["label"])
    print("Data:", first_class1["date"])
    print("Numero parole:", first_class1["num_words"])
    print("Primi 300 caratteri:")
    print(first_class1["text"][:300])
    print("Ultimi 300 caratteri:")
    print(first_class1["text"][-300:])

    print()
    print("=== ESEMPIO CLASS2 ===")

    first_class2 = clean_class2[0]

    print("ID:", first_class2["doc_id"])
    print("Label:", first_class2["label"])
    print("Data:", first_class2["date"])
    print("Numero parole:", first_class2["num_words"])
    print("Primi 300 caratteri:")
    print(first_class2["text"][:300])
    print("Ultimi 300 caratteri:")
    print(first_class2["text"][-300:])


if __name__ == "__main__":
    main()