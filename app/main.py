import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.chunking import chunk_log_text
from app.retrieval import IncidentRetriever
from app.context_builder import build_context
from app.investigator import RootCauseInvestigator
from app.evaluation import evaluate_report

from app.storage import (
    save_document,
    save_log_chunks,
    get_log_chunks,
    save_incident,
    save_investigation_report,
    save_incident_evaluation
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen2.5:3b"

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_CONTEXT_TOKENS = 2500


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RootCause AI",
    description=(
        "AI-powered incident investigation using "
        "hybrid RAG, Ollama and evidence validation."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class InvestigationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    log_text: str = Field(min_length=1, max_length=MAX_LOG_SIZE)


class EvaluationRequest(BaseModel):
    report: dict
    evidence: list[dict]
    context: str = Field(min_length=1)


# --------------------------------------------------
# SHARED INVESTIGATION PIPELINE
# --------------------------------------------------


def run_investigation(
    log_text: str,
    query: str,
    source: str
) -> dict:
    """
    Run the complete RootCause AI pipeline with MySQL persistence.

    1. Save or reuse document
    2. Save or reuse log chunks
    3. Load cached embeddings through IncidentRetriever
    4. Hybrid retrieval
    5. Context construction
    6. LLM investigation
    7. Automatic evaluation
    8. Persist incident, report and evaluation
    """

    if not log_text.strip():
        raise ValueError("Log content cannot be empty.")

    # --------------------------------------------------
    # 1. SAVE OR REUSE DOCUMENT
    # --------------------------------------------------

    document_result = save_document(
        file_name=source,
        content=log_text,
        source_type="LOG"
    )

    document_id = document_result["document_id"]

    print(
        f"Document ID: {document_id} | "
        f"Duplicate: {document_result['duplicate']}"
    )

    # --------------------------------------------------
    # 2. LOAD OR GENERATE LOG CHUNKS
    # --------------------------------------------------

    chunks = get_log_chunks(document_id)

    if not chunks:

        print("No stored chunks found. Generating new chunks...")

        generated_chunks = chunk_log_text(
            log_text,
            source=source,
            target_size=500,
            max_size=1000
        )

        if not generated_chunks:
            raise ValueError(
                "No usable chunks could be created."
            )

        save_log_chunks(
            document_id,
            generated_chunks
        )

        chunks = get_log_chunks(document_id)

    if not chunks:
        raise ValueError(
            "Unable to load or create document chunks."
        )

    print(f"Loaded {len(chunks)} chunks from MySQL.")

    # --------------------------------------------------
    # 3. INITIALIZE RETRIEVER WITH CACHED EMBEDDINGS
    # --------------------------------------------------

    retriever = IncidentRetriever(chunks)

    # --------------------------------------------------
    # 4. HYBRID RETRIEVAL
    # --------------------------------------------------

    retrieved_chunks = retriever.hybrid_search(
        query,
        top_k=3,
        candidate_k=5
    )

    if not retrieved_chunks:
        raise ValueError(
            "No relevant evidence was retrieved."
        )

    # --------------------------------------------------
    # 5. CONTEXT CONSTRUCTION
    # --------------------------------------------------

    context_result = build_context(
        retrieved_chunks,
        max_tokens=MAX_CONTEXT_TOKENS,
        source_text=log_text
    )

    context = context_result["context"]
    evidence = context_result["evidence"]

    if not evidence:
        raise ValueError(
            "No evidence available for investigation."
        )

    # --------------------------------------------------
    # 6. LLM INVESTIGATION
    # --------------------------------------------------

    investigator = RootCauseInvestigator(
        model=GENERATION_MODEL,
        temperature=0.2,
        max_tokens=800
    )

    report = investigator.investigate(
        query=query,
        context=context
    )

    # --------------------------------------------------
    # 7. AUTOMATIC EVALUATION
    # --------------------------------------------------

    evaluation = evaluate_report(
        report=report,
        evidence=evidence,
        context=context
    )

    # --------------------------------------------------
    # 8. PERSIST INVESTIGATION RESULTS
    # --------------------------------------------------

    incident_id = save_incident(
        document_id=document_id,
        query=query
    )

    report_id = save_investigation_report(
        incident_id=incident_id,
        report=report,
        model_name=GENERATION_MODEL
    )

    evaluation_result = save_incident_evaluation(
        report_id=report_id,
        evaluation=evaluation
    )

    # --------------------------------------------------
    # 9. RETURN COMPLETE RESULT
    # --------------------------------------------------

    return {
        "document_id": document_id,
        "incident_id": incident_id,
        "report_id": report_id,
        "query": query,
        "source": source,

        "statistics": {
            "total_chunks": len(chunks),
            "retrieved_chunks": len(retrieved_chunks),
            "evidence_count": context_result["evidence_count"],
            "estimated_context_tokens": (
                context_result["estimated_tokens"]
            )
        },

        "investigation": report,
        "evidence": evidence,
        "evaluation": evaluation,

        "persistence": {
            "document_saved": True,
            "duplicate_document": document_result["duplicate"],
            "chunks_saved": len(chunks),
            "evaluation_id": evaluation_result["evaluation_id"],
            "evaluation_status": evaluation_result["overall_status"]
        }
    }



# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "RootCause AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/investigate",
            "/evaluate",
            "/docs"
        ]
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        available_models = [
            model.get("name")
            for model in models
        ]

        embedding_available = any(
            name and name.startswith("nomic-embed-text")
            for name in available_models
        )

        generation_available = any(
            name and name.startswith("qwen2.5:3b")
            for name in available_models
        )

        return {
            "api": "running",
            "ollama": "connected",
            "models": {
                "embedding_model": {
                    "name": "nomic-embed-text",
                    "available": embedding_available
                },
                "generation_model": {
                    "name": GENERATION_MODEL,
                    "available": generation_available
                }
            },
            "ready": (
                embedding_available and generation_available
            )
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Start Ollama and retry."
        )


# --------------------------------------------------
# INVESTIGATION - RAW TEXT
# --------------------------------------------------

@app.post("/investigate")
def investigate_text(request: InvestigationRequest):

    try:
        return run_investigation(
            log_text=request.log_text,
            query=request.query,
            source="raw_text"
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable."
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


# --------------------------------------------------
# INVESTIGATION - LOG FILE UPLOAD
# --------------------------------------------------

@app.post("/investigate/file")
async def investigate_file(
    query: str = Form(...),
    file: UploadFile = File(...)
):

    if not query.strip():
        raise HTTPException(
            status_code=422,
            detail="Investigation query cannot be empty."
        )

    if not file.filename or not file.filename.lower().endswith(".log"):
        raise HTTPException(
            status_code=415,
            detail="Only .log files are supported."
        )

    try:
        content = await file.read(MAX_LOG_SIZE + 1)

        if len(content) > MAX_LOG_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Log file exceeds the 5 MB limit."
            )

        if not content:
            raise HTTPException(
                status_code=422,
                detail="Uploaded log file is empty."
            )

        log_text = content.decode(
            "utf-8",
            errors="replace"
        )

        return run_investigation(
            log_text=log_text,
            query=query,
            source=file.filename
        )

    except HTTPException:
        raise

    except UnicodeError:
        raise HTTPException(
            status_code=422,
            detail="Unable to decode the uploaded log file."
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable."
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    finally:
        await file.close()


# --------------------------------------------------
# EVALUATION ENDPOINT
# --------------------------------------------------

@app.post("/evaluate")
def evaluate_existing_report(request: EvaluationRequest):

    return evaluate_report(
        report=request.report,
        evidence=request.evidence,
        context=request.context
    )
