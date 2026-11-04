
from app.historical import generate_historical_embeddings


def test_historical_embeddings():

    result = generate_historical_embeddings()

    print("\nHistorical Embedding Results:")
    print(result)

    assert result["total_incidents"] >= 0

    print("\nHISTORICAL EMBEDDING TEST COMPLETED!")


if __name__ == "__main__":
    test_historical_embeddings()
