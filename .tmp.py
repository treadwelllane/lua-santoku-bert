from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2

model = SentenceTransformer("all-MiniLM-L6-v2")
index = IndexFlatL2(384)

sentences = ["This framework generates embeddings for each input sentence",
             "Sentences are passed as a list of string.",
             "The quick brown fox jumps over the lazy dog."]

for sentence in sentences:
    embd = model.encode(
            sentence,
            convert_to_numpy = True,
            normalize_embeddings = True).reshape(1, 384)
    index.add(embd)

query = model.encode(
        "the lazy dog",
        convert_to_numpy = True,
        normalize_embeddings = True).reshape(1, 384)

print(index.search(query, 3))
