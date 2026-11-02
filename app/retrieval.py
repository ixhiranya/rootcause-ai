
import math
import re
import requests

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Plus


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

stemmer = PorterStemmer()

STOP_WORDS = {
    "the", "is", "are", "a", "an",
    "why", "what", "how", "does",
    "do", "did", "in", "on", "at",
    "to", "for", "of", "and", "or",
    "it", "our", "we", "was", "were",
    "be", "been", "being", "this", "that"
}


# --------------------------------------------------
# TEXT TOKENIZATION
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Normalize text for keyword retrieval.

    Steps:
    1. Split camelCase/PascalCase words.
    2. Convert to lowercase.
    3. Extract alphanumeric tokens.
    4. Remove stop words.
    5. Apply Porter stemming.
    """

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    normalized_words = []

    for word in words:
        if word in STOP_WORDS:
            continue

        normalized_words.append(stemmer.stem(word))

    return normalized_words


# --------------------------------------------------
# OLLAMA EMBEDDINGS
# --------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using Ollama.

    Model: nomic-embed-text
    Endpoint: /api/embed
    """

    if not texts:
        return []

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": texts
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings, "
                f"received {len(embeddings)}"
            )

        return embeddings

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not generate embeddings using Ollama: {error}. "
            "Check whether Ollama is running."
        ) from error


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# --------------------------------------------------
# INCIDENT RETRIEVER
# --------------------------------------------------

class IncidentRetriever:
    """
    Hybrid incident retrieval system.

    Retrieval strategies:
    1. BM25Plus keyword retrieval.
    2. Semantic retrieval using Ollama embeddings.
    3. Hybrid retrieval using Reciprocal Rank Fusion.
    """

    def __init__(self, chunks: list[dict]):

        if not chunks:
            raise ValueError(
                "Cannot initialize retriever with empty chunks."
            )

        self.chunks = chunks

        # ------------------------------
        # Initialize keyword retrieval
        # ------------------------------

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Plus(self.tokenized_chunks)

        # ------------------------------
        # Initialize semantic retrieval
        # ------------------------------

        print("Generating document embeddings...")

        self.document_embeddings = get_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        print(
            f"Generated {len(self.document_embeddings)} embeddings."
        )

    # --------------------------------------------------
    # KEYWORD SEARCH - BM25Plus
    # --------------------------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = []

        for index, score in enumerate(scores):

            # BM25Plus can assign a baseline score to
            # documents without matching query terms.
            # Require at least one actual token match.
            matching_tokens = set(query_tokens).intersection(
                self.tokenized_chunks[index]
            )

            if not matching_tokens:
                continue

            result = self.chunks[index].copy()

            result["score"] = float(score)
            result["retrieval_method"] = "keyword"
            result["matched_tokens"] = sorted(matching_tokens)

            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        query = query.strip()

        if not query:
            return []

        query_embedding = get_embeddings([query])[0]

        scored_results = []

        for index, document_embedding in enumerate(
            self.document_embeddings
        ):

            score = cosine_similarity(
                query_embedding,
                document_embedding
            )

            result = self.chunks[index].copy()

            result["score"] = float(score)
            result["retrieval_method"] = "semantic"

            scored_results.append(result)

        scored_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return scored_results[:top_k]

    # --------------------------------------------------
    # HYBRID SEARCH - RECIPROCAL RANK FUSION
    # --------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
        rrf_k: int = 60
    ) -> list[dict]:
        """
        Combine BM25Plus and semantic rankings using RRF.

        RRF formula:

        RRF Score = 1 / (k + rank)

        Documents appearing near the top of both rankings
        receive higher combined scores.
        """

        keyword_results = self.keyword_search(
            query,
            top_k=candidate_k
        )

        semantic_results = self.semantic_search(
            query,
            top_k=candidate_k
        )

        fused_scores = {}
        chunk_lookup = {}

        # ------------------------------
        # Add keyword rankings
        # ------------------------------

        for rank, result in enumerate(keyword_results, start=1):

            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            chunk_lookup[chunk_id] = result

        # ------------------------------
        # Add semantic rankings
        # ------------------------------

        for rank, result in enumerate(semantic_results, start=1):

            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            chunk_lookup[chunk_id] = result

        # ------------------------------
        # Sort fused results
        # ------------------------------

        sorted_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True
        )

        final_results = []

        for chunk_id in sorted_chunk_ids[:top_k]:

            result = chunk_lookup[chunk_id].copy()

            result["score"] = fused_scores[chunk_id]
            result["retrieval_method"] = "hybrid"

            final_results.append(result)

        return final_results
