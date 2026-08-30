def load_faq(filepath):
    """Return the raw FAQ text loaded from the provided path."""

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_faq(text):
    """
    Splits FAQ text into chunks where each chunk
    contains one Question + Answer block.
    """

    normalized = text.replace("\r\n", "\n").strip()
    raw_chunks = normalized.split("\n\n")

    chunks = []
    for block in raw_chunks:
        chunk = block.strip()
        if chunk:
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    filepath = "data/faq.txt"
    text = load_faq(filepath)
    chunks = chunk_faq(text)

    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\nSample chunk:\n")
        print(chunks[0])
