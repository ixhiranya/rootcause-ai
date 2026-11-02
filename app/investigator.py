
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
4. If evidence is insufficient, explicitly say so.
5. Do not claim that a hypothesis is confirmed without
   sufficient evidence.
6. Recommend practical debugging steps.
7. Do not assume that a reachable database server means
   database connection acquisition is healthy.

Return ONLY valid JSON using this exact structure:

{
  "incident_summary": "Short incident summary",
  "probable_root_cause": "Most likely explanation",
  "confidence": "high, medium, or low",
  "supporting_evidence": [
    {
      "evidence_id": "E1",
      "observation": "What the evidence demonstrates"
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
  "missing_information": [
    "Information needed to confirm the root cause"
  ]
}

Use only evidence IDs that exist in the supplied context.
If no evidence supports a conclusion, state that clearly.
"""

        user_prompt = f"""
INCIDENT QUERY:
{query}

RETRIEVED INCIDENT EVIDENCE:
{context}

Analyze the incident and return the JSON report.
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

            if report["confidence"].lower() not in valid_confidence:
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
