
from app.retrieval import (
    get_embeddings,
    cosine_similarity,
    EMBEDDING_MODEL
)

from app.storage import (
    get_passed_historical_incidents,
    get_incident_embedding,
    save_incident_embedding
)


def build_historical_incident_text(incident: dict) -> str:
    """
    Combine useful historical incident information
    into one text for embedding generation.
    """

    report = incident["REPORT_JSON"]

    recommendations = report.get(
        "recommended_actions", []
    )

    recommendation_text = "\n".join(
        recommendations
    )

    return f"""
Historical Incident Query:
{incident["QUERY_TEXT"]}

Incident Summary:
{report.get("incident_summary", "")}

Probable Root Cause:
{report.get("probable_root_cause", "")}

Recommended Actions:
{recommendation_text}
""".strip()


def generate_historical_embeddings():
    """
    Generate and persist embeddings for all
    successfully evaluated historical incidents.
    """

    incidents = get_passed_historical_incidents()

    if not incidents:
        print("No passed historical incidents found.")
        return {
            "total_incidents": 0,
            "new_embeddings": 0,
            "cached_embeddings": 0
        }

    missing_incidents = []
    missing_texts = []

    cached_count = 0

    for incident in incidents:

        cached_embedding = get_incident_embedding(
            incident["INCIDENT_ID"],
            EMBEDDING_MODEL
        )

        if (
            cached_embedding is not None
            and len(cached_embedding) == 768
        ):
            cached_count += 1

            print(
                f"Using cached embedding for incident "
                f"{incident['INCIDENT_ID']}"
            )

            continue

        incident_text = build_historical_incident_text(
            incident
        )

        missing_incidents.append(incident)
        missing_texts.append(incident_text)

    new_count = 0

    if missing_texts:

        embeddings = get_embeddings(missing_texts)

        for incident, embedding in zip(
            missing_incidents,
            embeddings
        ):

            save_incident_embedding(
                incident["INCIDENT_ID"],
                EMBEDDING_MODEL,
                embedding
            )

            new_count += 1

            print(
                f"Saved historical embedding for incident "
                f"{incident['INCIDENT_ID']}"
            )

    return {
        "total_incidents": len(incidents),
        "new_embeddings": new_count,
        "cached_embeddings": cached_count
    }

def search_similar_historical_incidents(
    query: str,
    context: str = "",
    top_k: int = 3
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Historical search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    search_text = f"""
    Incident Query:
    {query}

    Current Incident Evidence:
    {context}
    """.strip()

    query_embedding = get_embeddings([search_text])[0]

    historical_incidents = get_passed_historical_incidents()

    if not historical_incidents:
        return []

    scored_incidents = []

    for incident in historical_incidents:

        incident_id = incident["INCIDENT_ID"]

        historical_embedding = get_incident_embedding(
            incident_id,
            EMBEDDING_MODEL
        )

        if historical_embedding is None:
            continue

        if len(historical_embedding) != len(query_embedding):
            continue

        similarity = cosine_similarity(
            query_embedding,
            historical_embedding
        )

        report = incident["REPORT_JSON"]

        scored_incidents.append({
            "incident_id": incident_id,
            "report_id": incident["REPORT_ID"],
            "query": incident["QUERY_TEXT"],
            "incident_summary": report.get("incident_summary", ""),
            "probable_root_cause": report.get("probable_root_cause", ""),
            "recommended_actions": report.get("recommended_actions", []),
            "confidence": report.get("confidence", "unknown"),
            "evaluation_status": "PASSED",
            "similarity_score": float(similarity)
        })

    scored_incidents.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    return scored_incidents[:top_k]
