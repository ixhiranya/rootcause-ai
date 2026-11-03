
import hashlib
import json
from app.database import get_db_connection



def calculate_file_hash(content: str) -> str:
    """Generate a SHA-256 hash for document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_document_by_hash(file_hash: str):
    """Find an existing document using its content hash."""
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT DOCUMENT_ID, FILE_NAME, SOURCE_TYPE, FILE_HASH, CREATED_AT
            FROM DOCUMENTS
            WHERE FILE_HASH = %s
            ORDER BY DOCUMENT_ID
            LIMIT 1
            """,
            (file_hash,)
        )

        document = cursor.fetchone()
        cursor.close()

        return document

    finally:
        if connection and connection.is_connected():
            connection.close()


def save_document(file_name: str, content: str, source_type: str = "LOG"):
    """
    Save a document if it does not already exist.

    Returns:
        DOCUMENT_ID
        Whether the document was newly created
        Whether an existing document was reused
    """
    if not content or not content.strip():
        raise ValueError("Document content cannot be empty.")

    file_hash = calculate_file_hash(content)

    # Check whether this document already exists.
    existing_document = get_document_by_hash(file_hash)

    if existing_document:
        return {
            "document_id": existing_document["DOCUMENT_ID"],
            "file_name": existing_document["FILE_NAME"],
            "created": False,
            "duplicate": True,
            "file_hash": file_hash
        }

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO DOCUMENTS (
                FILE_NAME,
                SOURCE_TYPE,
                FILE_HASH,
                DOCUMENT_CONTENT
            )
            VALUES (%s, %s, %s, %s)
            """,
            (file_name, source_type, file_hash, content)
        )

        document_id = cursor.lastrowid
        connection.commit()
        cursor.close()

        return {
            "document_id": document_id,
            "file_name": file_name,
            "created": True,
            "duplicate": False,
            "file_hash": file_hash
        }

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()
    

from app.chunking import chunk_log_text


def save_log_chunks(document_id: int, chunks: list):
    """Persist generated log chunks for a document."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        saved_count = 0

        for chunk in chunks:
            cursor.execute(
                """
                INSERT INTO LOG_CHUNKS (
                    DOCUMENT_ID,
                    CHUNK_KEY,
                    CHUNK_INDEX,
                    CHUNK_TEXT,
                    START_CHAR,
                    END_CHAR
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    CHUNK_TEXT = VALUES(CHUNK_TEXT),
                    START_CHAR = VALUES(START_CHAR),
                    END_CHAR = VALUES(END_CHAR)
                """,
                (
                    document_id,
                    chunk["chunk_id"],
                    chunk["chunk_index"],
                    chunk["text"],
                    chunk["start_char"],
                    chunk["end_char"]
                )
            )

            saved_count += 1

        connection.commit()

        return {
            "document_id": document_id,
            "chunks_saved": saved_count
        }

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()



def get_log_chunks(document_id: int):
    """Retrieve persisted chunks with their source metadata."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                LC.CHUNK_ID,
                LC.DOCUMENT_ID,
                LC.CHUNK_KEY,
                LC.CHUNK_INDEX,
                LC.CHUNK_TEXT,
                LC.START_CHAR,
                LC.END_CHAR,
                D.FILE_NAME,
                D.SOURCE_TYPE
            FROM LOG_CHUNKS LC
            INNER JOIN DOCUMENTS D
                ON D.DOCUMENT_ID = LC.DOCUMENT_ID
            WHERE LC.DOCUMENT_ID = %s
            ORDER BY LC.CHUNK_INDEX
            """,
            (document_id,)
        )

        chunks = cursor.fetchall()
        cursor.close()

        return [
            {
                "db_chunk_id": row["CHUNK_ID"],
                "chunk_id": row["CHUNK_KEY"],
                "chunk_index": row["CHUNK_INDEX"],
                "text": row["CHUNK_TEXT"],
                "start_char": row["START_CHAR"],
                "end_char": row["END_CHAR"],
                "source": row["FILE_NAME"],
                "source_type": row["SOURCE_TYPE"]
            }
            for row in chunks
        ]

    finally:
        if connection and connection.is_connected():
            connection.close()



def save_chunk_embedding(chunk_id: int, model_name: str, embedding: list):
    """Save or update an embedding for a database chunk."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        embedding_json = json.dumps(embedding)

        cursor.execute(
            """
            INSERT INTO CHUNK_EMBEDDINGS (
                CHUNK_ID,
                MODEL_NAME,
                EMBEDDING_VECTOR
            )
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                EMBEDDING_VECTOR = VALUES(EMBEDDING_VECTOR)
            """,
            (chunk_id, model_name, embedding_json)
        )

        connection.commit()
        cursor.close()

        return {
            "chunk_id": chunk_id,
            "model_name": model_name,
            "saved": True
        }

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()


def get_chunk_embedding(chunk_id: int, model_name: str):
    """Retrieve a cached embedding for a chunk and model."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT EMBEDDING_VECTOR
            FROM CHUNK_EMBEDDINGS
            WHERE CHUNK_ID = %s AND MODEL_NAME = %s
            LIMIT 1
            """,
            (chunk_id, model_name)
        )

        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        vector = row["EMBEDDING_VECTOR"]

        # MySQL JSON values are commonly returned as strings.
        if isinstance(vector, str):
            vector = json.loads(vector)

        return vector

    finally:
        if connection and connection.is_connected():
            connection.close()




def save_incident(document_id: int, query: str):
    """Create an incident record for an investigation."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO INCIDENTS (
                DOCUMENT_ID,
                QUERY_TEXT,
                STATUS
            )
            VALUES (%s, %s, %s)
            """,
            (document_id, query, "COMPLETED")
        )

        incident_id = cursor.lastrowid
        connection.commit()
        cursor.close()

        return incident_id

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()


def save_investigation_report(
    incident_id: int,
    report: dict,
    model_name: str
):
    """Persist the AI-generated investigation report."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO INVESTIGATION_REPORTS (
                INCIDENT_ID,
                REPORT_JSON,
                MODEL_NAME
            )
            VALUES (%s, %s, %s)
            """,
            (
                incident_id,
                json.dumps(report),
                model_name
            )
        )

        report_id = cursor.lastrowid
        connection.commit()
        cursor.close()

        return report_id

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()


def save_incident_evaluation(
    report_id: int,
    evaluation: dict
):
    """Persist evaluation results for an investigation report."""

    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        passed = evaluation.get(
            "overall_validation_passed",
            False
        )

        overall_status = "PASSED" if passed else "FAILED"

        cursor.execute(
            """
            INSERT INTO INCIDENT_EVALUATIONS (
                REPORT_ID,
                EVALUATION_JSON,
                OVERALL_STATUS
            )
            VALUES (%s, %s, %s)
            """,
            (
                report_id,
                json.dumps(evaluation),
                overall_status
            )
        )

        evaluation_id = cursor.lastrowid
        connection.commit()
        cursor.close()

        return {
            "evaluation_id": evaluation_id,
            "overall_status": overall_status
        }

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()
