
from app.historical import search_similar_historical_incidents


def test_historical_search():

    query = "Why are database order requests failing?"

    context = """
    Connection pool exhausted:
    maximum connections=10,
    active connections=10.

    Database query failed after 3000ms.
    Connection acquisition timed out.
    GET /api/orders returned HTTP 500.
    """

    results = search_similar_historical_incidents(
        query=query,
        context=context,
        top_k=3
    )

    print("\nHistorical Search Results:")

    for result in results:
        print("\nIncident ID:", result["incident_id"])
        print("Similarity:", result["similarity_score"])
        print("Root Cause:", result["probable_root_cause"])
        print("Evaluation:", result["evaluation_status"])

    assert len(results) <= 3

    for result in results:
        assert result["evaluation_status"] == "PASSED"
        assert  result["similarity_score"] <= 1.0
        assert result["similarity_score"] >= -1.0

    print("\nHISTORICAL SEARCH TEST COMPLETED!")


if __name__ == "__main__":
    test_historical_search()
