import os
import math
import re
from collections import Counter, defaultdict

# ===================== Tokenizer =====================

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

# ===================== Load Documents =====================

def load_documents(folder):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs[filename] = tokenize(f.read())
    return docs

# ===================== TF =====================

def compute_tf(tokens):
    tf = Counter(tokens)
    total = len(tokens)
    return {term: freq / total for term, freq in tf.items()}

# ===================== IDF =====================

def compute_idf(docs):
    N = len(docs)
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1
    return {term: math.log(N / df[term]) for term in df}

# ===================== TF-IDF =====================

def compute_tfidf(docs, idf):
    tfidf_docs = {}
    for doc, tokens in docs.items():
        tf = compute_tf(tokens)
        tfidf_docs[doc] = {
            term: tf[term] * idf[term]
            for term in tf
        }
    return tfidf_docs

# ===================== Cosine Similarity =====================

def cosine_similarity(vec1, vec2):
    common = set(vec1) & set(vec2)
    numerator = sum(vec1[t] * vec2[t] for t in common)
    denom1 = math.sqrt(sum(v*v for v in vec1.values()))
    denom2 = math.sqrt(sum(v*v for v in vec2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / (denom1 * denom2)

# ===================== Query =====================

def process_query(query, idf):
    tokens = tokenize(query)
    tf = compute_tf(tokens)
    return {
        term: tf[term] * idf.get(term, 0)
        for term in tf
    }

# ===================== Main =====================

if __name__ == "__main__":
    docs = load_documents("docs")

    print("Documents loaded:")
    for d in docs:
        print(" -", d)

    idf = compute_idf(docs)
    tfidf_docs = compute_tfidf(docs, idf)

    query = input("\nEnter your query: ")
    query_vec = process_query(query, idf)

    scores = []
    for doc, vec in tfidf_docs.items():
        score = cosine_similarity(query_vec, vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Search Results ===")
    for doc, score in scores:
        print(f"{doc} -> similarity = {score:.4f}")
