
from app.database import get_db_connection
from app.storage import (
    save_document,
    save_incident,
    save_investigation_report,
    save_incident_evaluation
)
from app.evaluation import evaluate_report
from app.historical import generate_historical_embeddings


# --------------------------------------------------
# SYNTHETIC HISTORICAL INCIDENT DATASET
# --------------------------------------------------

TEST_INCIDENTS = [
    {
        "category": "authentication",
        "query": "Why are users receiving unauthorized errors?",
        "log": """2026-11-01 09:10:01 ERROR JWT token validation failed
2026-11-01 09:10:02 ERROR Token expired for user_id=1024
2026-11-01 09:10:03 ERROR GET /api/profile returned HTTP 401""",
        "summary": "Users cannot access protected API endpoints because JWT tokens have expired.",
        "root_cause": "Expired JWT tokens are failing authentication validation, resulting in HTTP 401 responses.",
        "actions": [
            "Check JWT token expiration configuration.",
            "Verify the token refresh mechanism.",
            "Review authentication middleware and client token renewal behavior."
        ]
    },
    {
        "category": "api_timeout",
        "query": "Why are API requests failing with gateway timeout errors?",
        "log": """2026-11-01 10:20:01 ERROR Upstream service request timed out
2026-11-01 10:20:02 ERROR Gateway timeout after 30000ms
2026-11-01 10:20:03 ERROR GET /api/orders returned HTTP 504""",
        "summary": "API requests are failing because the upstream service does not respond within the gateway timeout period.",
        "root_cause": "The upstream service response exceeds the gateway timeout limit, resulting in HTTP 504 errors.",
        "actions": [
            "Inspect upstream service response latency.",
            "Check gateway timeout configuration.",
            "Review slow database queries and downstream dependencies."
        ]
    },
    {
        "category": "memory",
        "query": "Why is the application crashing due to memory exhaustion?",
        "log": """2026-11-01 11:30:01 ERROR Java heap space exhausted
2026-11-01 11:30:02 ERROR OutOfMemoryError: Java heap space
2026-11-01 11:30:03 ERROR Application process terminated unexpectedly""",
        "summary": "The application crashes after exhausting the available Java heap memory.",
        "root_cause": "The Java application has exhausted its configured heap memory, causing an OutOfMemoryError.",
        "actions": [
            "Capture and analyze a heap dump.",
            "Inspect memory usage and object retention.",
            "Check for memory leaks and unbounded collections."
        ]
    },
    {
        "category": "network",
        "query": "Why is the application unable to connect to the remote server?",
        "log": """2026-11-01 12:40:01 ERROR DNS resolution failed for api.example.internal
2026-11-01 12:40:02 ERROR NameResolutionError: failed to resolve hostname
2026-11-01 12:40:03 ERROR Remote API connection could not be established""",
        "summary": "The application cannot connect to a remote API because hostname resolution fails.",
        "root_cause": "DNS resolution failure prevents the application from resolving the remote server hostname.",
        "actions": [
            "Verify DNS server availability and configuration.",
            "Check hostname and DNS records.",
            "Inspect network connectivity from the application host."
        ]
    }
]


# --------------------------------------------------
# CHECK FOR EXISTING TEST INCIDENT
# --------------------------------------------------

def find_existing_incident(document_id, query):

    connection = get_db_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT INCIDENT_ID
            FROM INCIDENTS
            WHERE DOCUMENT_ID = %s
              AND QUERY_TEXT = %s
            ORDER BY INCIDENT_ID
            LIMIT 1
            """,
            (document_id, query)
        )

        result = cursor.fetchone()
        cursor.close()

        return result["INCIDENT_ID"] if result else None

    finally:
        if connection.is_connected():
            connection.close()


# --------------------------------------------------
# SEED HISTORICAL INCIDENTS
# --------------------------------------------------

def seed_historical_incidents():

    created_incidents = []

    print("\nROOTCAUSE AI - HISTORICAL DATASET SEEDING")
    print("=" * 55)

    for item in TEST_INCIDENTS:

        print(f"\nProcessing: {item['category']}")

        document_result = save_document(
            file_name=f"synthetic_{item['category']}.log",
            content=item["log"],
            source_type="TEST_LOG"
        )

        document_id = document_result["document_id"]

        existing_id = find_existing_incident(
            document_id,
            item["query"]
        )

        if existing_id:
            print(f"Already exists. Incident ID: {existing_id}")
            created_incidents.append({
                "category": item["category"],
                "incident_id": existing_id
            })
            continue

        incident_id = save_incident(
            document_id=document_id,
            query=item["query"]
        )

        evidence = [{
            "evidence_id": "E1",
            "text": item["log"]
        }]

        report = {
            "incident_summary": item["summary"],
            "probable_root_cause": item["root_cause"],
            "confidence": "high",
            "supporting_evidence": [{
                "evidence_id": "E1",
                "start_line": 1,
                "end_line": 3,
                "observation": item["log"]
            }],
            "alternative_hypotheses": [],
            "recommended_actions": item["actions"],
            "missing_information": []
        }

        evaluation = evaluate_report(
            report=report,
            evidence=evidence,
            context=item["log"]
        )

        if not evaluation["overall_validation_passed"]:
            raise RuntimeError(
                f"Evaluation failed for {item['category']}: "
                f"{evaluation}"
            )

        report_id = save_investigation_report(
            incident_id=incident_id,
            report=report,
            model_name="synthetic-test-data"
        )

        evaluation_result = save_incident_evaluation(
            report_id=report_id,
            evaluation=evaluation
        )

        if evaluation_result["overall_status"] != "PASSED":
            raise RuntimeError(
                f"Incident {incident_id} did not pass evaluation."
            )

        created_incidents.append({
            "category": item["category"],
            "incident_id": incident_id,
            "report_id": report_id,
            "evaluation_status": evaluation_result["overall_status"]
        })

        print(f"Incident ID: {incident_id}")
        print(f"Report ID: {report_id}")
        print("Evaluation: PASSED")

    print("\nDATASET SEEDING SUMMARY")
    print("=" * 55)

    for item in created_incidents:
        print(item)

    print("\nGenerating historical embeddings...")

    embedding_results = generate_historical_embeddings()

    print("\nEMBEDDING RESULTS")
    print(embedding_results)

    print("\nHISTORICAL DATASET SEEDING COMPLETED!")

    return created_incidents


if __name__ == "__main__":
    seed_historical_incidents()
