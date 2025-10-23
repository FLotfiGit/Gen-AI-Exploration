# Minimal smoke test for TFIDF utilities in coding_camp.py

from coding_camp import TFIDF


docs = [
    "cats like milk",
    "dogs play outside",
    "milk is healthy",
    "the sky is blue",
]


def test_query_matches_relevant_doc():
    tfidf = TFIDF()
    tfidf.fit(docs)
    corpus_vecs = tfidf.transform(docs)

    query = "cats drink milk"
    query_vec = tfidf.transform([query])[0]

    # Compute cosine similarities to each doc in the corpus
    scores = [tfidf.cosine_sparse(query_vec, d_vec) for d_vec in corpus_vecs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    # Expect the first document ("cats like milk") to be the most similar
    assert best_idx == 0, f"Expected doc index 0 to be most similar, got {best_idx} with scores {scores}"