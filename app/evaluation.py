
import re


# --------------------------------------------------
# 1. RETRIEVAL EVALUATION
# --------------------------------------------------

def evaluate_retrieval(
    retrieved_chunks: list[dict],
    relevant_chunk_ids: list[str],
    k: int = 3
) -> dict:

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_chunk_ids:
        raise ValueError(
            "At least one relevant chunk ID is required."
        )

    relevant_ids = set(relevant_chunk_ids)

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks[:k]
    ]

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    ]

    recall_at_k = (
        len(set(relevant_retrieved)) / len(relevant_ids)
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(reciprocal_rank, 4),
        "relevant_chunks_found": len(set(relevant_retrieved)),
        "total_relevant_chunks": len(relevant_ids),
        "retrieved_chunk_ids": retrieved_ids
    }


# --------------------------------------------------
# 2. CITATION VALIDATION
# --------------------------------------------------

def evaluate_citations(
    report: dict,
    evidence: list[dict]
) -> dict:

    valid_evidence_ids = {
        item["evidence_id"]
        for item in evidence
    }

    cited_ids = []

    for item in report.get("supporting_evidence", []):
        evidence_id = item.get("evidence_id")

        if evidence_id:
            cited_ids.append(evidence_id)

    valid_citations = [
        evidence_id
        for evidence_id in cited_ids
        if evidence_id in valid_evidence_ids
    ]

    invalid_citations = [
        evidence_id
        for evidence_id in cited_ids
        if evidence_id not in valid_evidence_ids
    ]

    total_citations = len(cited_ids)

    accuracy = (
        len(valid_citations) / total_citations
        if total_citations > 0
        else 0.0
    )

    return {
        "citation_accuracy": round(accuracy, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "all_citations_valid": (
            total_citations > 0
            and len(invalid_citations) == 0
        )
    }


# --------------------------------------------------
# 3. KNOWN FACT DEFINITIONS
# --------------------------------------------------

KNOWN_FACTS = {
    "maximum connections": {
        "context_patterns": [
            r"\bmaximum\s+connections?\s*[=:]\s*\d+\b",
            r"\bmax\s+connections?\s*[=:]\s*\d+\b",
            r"\bconnection\s+pool\s+size\s*[=:]\s*\d+\b"
        ],
        "claim_patterns": [
            r"\bmaximum\s+connections?\b",
            r"\bmax\s+connections?\b",
            r"\bconnection\s+pool\s+size\b",
            r"\bpool\s+size\b"
        ]
    },

    "active connections": {
        "context_patterns": [
            r"\bactive\s+connections?\s*[=:]\s*\d+\b",
            r"\bcurrent(?:ly)?\s+active\s+connections?\s*[=:]\s*\d+\b"
        ],
        "claim_patterns": [
            r"\bactive\s+connections?\b",
            r"\bcurrent\s+number\s+of\s+active\s+connections?\b",
            r"\bcurrently\s+active\s+connections?\b"
        ]
    },

    "http status": {
        "context_patterns": [
            r"\bhttp\s*500\b",
            r"\bhttp\s+status\s*[=:]\s*500\b",
            r"\bstatus\s+code\s*[=:]\s*500\b"
        ],
        "claim_patterns": [
            r"\bhttp\s+500\b",
            r"\bhttp\s+status\b",
            r"\bstatus\s+code\b",
            r"\bresponse\s+status\b"
        ]
    }
}


# --------------------------------------------------
# 4. EVIDENCE CONSISTENCY CHECK
# --------------------------------------------------

def check_missing_information(
    report: dict,
    context: str
) -> dict:
    """
    Detect when the model claims information is missing
    even though the evidence context contains that fact.

    Uses fact aliases rather than requiring exact wording.
    """

    normalized_context = re.sub(
        r"\s+",
        " ",
        context.lower()
    )

    detected_facts = []
    detected_fact_values = {}

    # Identify facts available in the evidence.
    for fact_name, patterns in KNOWN_FACTS.items():

        matched_pattern = None

        for pattern in patterns["context_patterns"]:

            match = re.search(
                pattern,
                normalized_context
            )

            if match:
                matched_pattern = match
                break

        if matched_pattern:

            detected_facts.append(fact_name)

            # Capture the numerical value when available.
            number_match = re.search(
                r"\d+",
                matched_pattern.group()
            )

            detected_fact_values[fact_name] = (
                int(number_match.group())
                if number_match
                else None
            )

    inconsistencies = []

    # Compare model's missing-information claims
    # against facts already present in the context.
    for missing_item in report.get("missing_information", []):

        normalized_item = re.sub(
            r"\s+",
            " ",
            missing_item.lower()
        )

        for fact_name in detected_facts:

            claim_patterns = KNOWN_FACTS[fact_name][
                "claim_patterns"
            ]

            claim_mentions_fact = any(
                re.search(pattern, normalized_item)
                for pattern in claim_patterns
            )

            if claim_mentions_fact:

                inconsistencies.append({
                    "model_claim": missing_item,
                    "fact_found_in_context": fact_name,
                    "known_value": detected_fact_values[fact_name],
                    "issue": (
                        "The model claims this information is missing, "
                        "but the evidence context already contains it."
                    )
                })

    return {
        "consistency_passed": len(inconsistencies) == 0,
        "detected_facts": detected_facts,
        "detected_fact_values": detected_fact_values,
        "inconsistencies": inconsistencies
    }


# --------------------------------------------------
# 5. COMBINED REPORT EVALUATION
# --------------------------------------------------

def evaluate_report(
    report: dict,
    evidence: list[dict],
    context: str
) -> dict:

    citation_results = evaluate_citations(
        report,
        evidence
    )

    consistency_results = check_missing_information(
        report,
        context
    )

    return {
        "citation_evaluation": citation_results,
        "consistency_evaluation": consistency_results,
        "overall_validation_passed": (
            citation_results["all_citations_valid"]
            and consistency_results["consistency_passed"]
        )
    }
