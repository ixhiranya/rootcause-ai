
import json
import requests


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen2.5:3b"


# --------------------------------------------------
# ROOT CAUSE INVESTIGATOR
# --------------------------------------------------

class RootCauseInvestigator:

    def __init__(
        self,
        model: str = GENERATION_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 800
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # --------------------------------------------------
    # BUILD INVESTIGATION PROMPT
    # --------------------------------------------------

    def build_prompt(
        self,
        query: str,
        context: str
    ) -> list[dict]:

        system_prompt = """
You are RootCause AI, an expert software debugging assistant.

Your task is to investigate application incidents using
only the evidence provided in the user's message.

IMPORTANT RULES:

1. Never invent log events, error messages, or system behavior.
2. Distinguish confirmed facts from possible explanations.
3. Every supporting evidence item must reference an actual
   evidence ID such as E1 or E2.
4. Do not claim that a hypothesis is confirmed without
   sufficient evidence.
5. Recommend practical debugging steps based on the evidence.
6. Do not assume that a reachable database server means
   database connection acquisition is healthy.

MISSING INFORMATION RULES — VERY IMPORTANT:

1. Before adding anything to missing_information, carefully
   inspect the entire retrieved evidence.

2. A fact must NOT be reported as missing if it is already
   explicitly available in the evidence, even if it is
   expressed using different wording.

3. Treat equivalent technical terms as the same information.
   For example:
   - "maximum connections=10" means the connection pool size
     or maximum pool capacity is 10.
   - "active connections=10" means the current number of
     active connections is 10.
   - "HTTP 500" means the request returned status code 500.

4. Do not list information as missing simply because the
   evidence uses a different technical term or wording.

5. Only include information that is genuinely absent and
   necessary to investigate or confirm the root cause.

6. If all relevant information is already available, return
   an EMPTY ARRAY for missing_information: [].

7. Do not repeat known facts in missing_information.

8. Distinguish between:
   - Known facts: explicitly present in the logs.
   - Unknown facts: not present in the supplied evidence.
   - Hypotheses: possible explanations that need verification.

ROOT CAUSE RULES:

1. Prefer explanations directly supported by log evidence.
2. Do not confuse a symptom with the underlying root cause.
3. Clearly identify uncertainty when multiple explanations
   are possible.
4. Do not recommend increasing configuration limits as the
   only solution without considering why the limit was reached.
5. Recommendations should include practical investigation
   steps wherever possible.

Return ONLY valid JSON using this exact structure:

{
  "incident_summary": "Short incident summary",
  "probable_root_cause": "Most likely explanation",
  "confidence": "high, medium, or low",
  "supporting_evidence": [
    {
      "evidence_id": "E1",
      "start_line": 3,
      "end_line": 5,
      "observation": "Database connection pool exhausted."
    }
  ],
  "alternative_hypotheses": [
    {
      "hypothesis": "Alternative possible explanation",
      "reason": "Why it may be relevant"
    }
  ],
  "recommended_actions": [
    "First debugging action",
    "Second debugging action"
  ],
  "missing_information": []
}

Use only evidence IDs that exist in the supplied context.
Use exact source line numbers when available.
Never invent line numbers.

If no evidence supports a conclusion, state that clearly.
"""

        user_prompt = f"""
INCIDENT QUERY:
{query}

RETRIEVED INCIDENT EVIDENCE:
{context}

Analyze the incident and return the JSON report.

EVIDENCE CITATION RULES:

1. Cite the evidence ID for every supporting observation.
2. Include the original source line range when available.
3. Never invent line numbers.
4. Use the exact start_line and end_line provided in evidence.
5. If line numbers are unavailable, cite only the evidence ID.

FINAL VERIFICATION BEFORE RESPONDING:

1. Review every item in missing_information.
2. Check whether the information is already present in the
   retrieved evidence, including equivalent terminology.
3. Remove any item that describes information already known.
4. If nothing genuinely necessary is missing, return [].
5. Ensure the response is valid JSON and follows the exact
   required structure.
"""

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    # --------------------------------------------------
    # GENERATE ROOT CAUSE REPORT
    # --------------------------------------------------

    def investigate(
        self,
        query: str,
        context: str
    ) -> dict:

        if not query.strip():
            raise ValueError("Investigation query cannot be empty.")

        if not context.strip():
            raise ValueError("Investigation context cannot be empty.")

        messages = self.build_prompt(query, context)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=180
            )

            response.raise_for_status()

            response_data = response.json()

            content = response_data.get(
                "message", {}
            ).get("content", "")

            if not content:
                raise ValueError(
                    "Ollama returned an empty response."
                )

            report = json.loads(content)

            # Basic output validation
            required_fields = [
                "incident_summary",
                "probable_root_cause",
                "confidence",
                "supporting_evidence",
                "alternative_hypotheses",
                "recommended_actions",
                "missing_information"
            ]

            for field in required_fields:
                if field not in report:
                    raise ValueError(
                        f"Missing report field: {field}"
                    )

            valid_confidence = {"high", "medium", "low"}

            if (
                not isinstance(report["confidence"], str)
                or report["confidence"].lower() not in valid_confidence
            ):
                raise ValueError("Invalid confidence value.")

            return report

        except requests.RequestException as error:
            raise RuntimeError(
                f"Ollama generation failed: {error}. "
                "Check whether Ollama is running."
            ) from error

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The model did not return valid JSON."
            ) from error
