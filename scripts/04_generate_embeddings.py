import os
from pathlib import Path

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_and_store_embeddings():

    # ---------------------------------------------------------
    # 1. Load environment variables
    # ---------------------------------------------------------

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    env_path = project_root / "data" / ".env"
    load_dotenv(env_path, override=True)

    # ---------------------------------------------------------
    # 2. Read database configuration
    # ---------------------------------------------------------

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([
        db_user,
        db_password,
        db_host,
        db_port,
        db_name
    ]):
        raise ValueError(
            "Database configuration is incomplete. "
            "Check DB_USER, DB_PASSWORD, DB_HOST, DB_PORT "
            "and DB_NAME in data/.env"
        )

    # ---------------------------------------------------------
    # 3. Build PostgreSQL connection URL
    # ---------------------------------------------------------

    db_url = (
        f"postgresql+psycopg2://"
        f"{db_user}:{db_password}@"
        f"{db_host}:{db_port}/{db_name}"
    )

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300
    )

    # ---------------------------------------------------------
    # 4. Load BioClinical ModernBERT embedding model
    # ---------------------------------------------------------

    print("Loading Local BioClinical ModernBERT model...")

    model = SentenceTransformer(
        "NeuML/bioclinical-modernbert-base-embeddings"
    )

    # ---------------------------------------------------------
    # 5. Verify embedding dimension
    # ---------------------------------------------------------

    embedding_dimension = (
        model.get_sentence_embedding_dimension()
    )

    print(
        f"Model embedding dimension: {embedding_dimension}"
    )

    if embedding_dimension != 768:
        raise ValueError(
            "CRITICAL DIMENSION MISMATCH: "
            f"Model output is {embedding_dimension}, "
            "but the database expects 768 dimensions."
        )

    print("Embedding dimension verified: 768")

    # ---------------------------------------------------------
    # 6. Processing configuration
    # ---------------------------------------------------------

    batch_size = 2

    # Process only 2 records for the first test
    demo_limit = 2

    # ---------------------------------------------------------
    # 7. Connect to PostgreSQL
    # ---------------------------------------------------------

    try:

        with engine.begin() as conn:

            print(
                f"\nGenerating batched embeddings for "
                f"up to {demo_limit} patient records..."
            )

            processed = 0

            # -------------------------------------------------
            # Progress bar
            # -------------------------------------------------

            with tqdm(
                total=demo_limit,
                desc="Vectorizing clinical records",
                unit="rows"
            ) as pbar:

                # -------------------------------------------------
                # 8. Process records until demo limit is reached
                # -------------------------------------------------

                while processed < demo_limit:

                    remaining = demo_limit - processed

                    current_batch_size = min(
                        batch_size,
                        remaining
                    )

                    # -------------------------------------------------
                    # 9. Fetch records without embeddings
                    # -------------------------------------------------

                    select_query = text(
                        """
                        SELECT
                            id,
                            admission_type,
                            drug,
                            test_name,
                            drg_severity,
                            description,
                            comments
                        FROM patient_encounters
                        WHERE clinical_embedding IS NULL
                        LIMIT :batch_size
                        """
                    )

                    batch = conn.execute(
                        select_query,
                        {
                            "batch_size": current_batch_size
                        }
                    ).mappings().fetchall()

                    if not batch:
                        print(
                            "\nNo more records found "
                            "without embeddings."
                        )
                        break

                    # -------------------------------------------------
                    # 10. Prepare clinical text
                    # -------------------------------------------------

                    clinical_texts = []
                    ids = []

                    for row in batch:

                        components = []

                        if row["admission_type"]:
                            components.append(
                                f"Admission: "
                                f"{row['admission_type']}"
                            )

                        if row["drug"]:
                            components.append(
                                f"Prescribed: "
                                f"{row['drug']}"
                            )

                        if row["test_name"]:
                            components.append(
                                f"Lab Test: "
                                f"{row['test_name']}"
                            )

                        if row["drg_severity"]:
                            components.append(
                                f"Severity level: "
                                f"{row['drg_severity']}"
                            )

                        if row["description"]:
                            components.append(
                                f"Diagnosis: "
                                f"{row['description']}"
                            )

                        if row["comments"]:
                            components.append(
                                f"Notes: "
                                f"{row['comments']}"
                            )

                        clinical_text = " | ".join(
                            components
                        )

                        if not clinical_text:
                            clinical_text = (
                                "No clinical information available."
                            )

                        clinical_texts.append(
                            clinical_text
                        )

                        ids.append(
                            row["id"]
                        )

                    # -------------------------------------------------
                    # 11. Generate embeddings
                    # -------------------------------------------------

                    embeddings = model.encode(
                        clinical_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )

                    # -------------------------------------------------
                    # 12. Prepare update parameters
                    # -------------------------------------------------

                    update_params = [
                        {
                            "id": record_id,
                            "embedding": (
                                "["
                                + ",".join(
                                    str(float(value))
                                    for value in embedding
                                )
                                + "]"
                            )
                        }
                        for record_id, embedding
                        in zip(ids, embeddings)
                    ]

                    # -------------------------------------------------
                    # 13. Store embeddings in PostgreSQL
                    # -------------------------------------------------

                    update_query = text(
                        """
                        UPDATE patient_encounters
                        SET clinical_embedding =
                            CAST(:embedding AS vector)
                        WHERE id = :id
                        """
                    )

                    conn.execute(
                        update_query,
                        update_params
                    )

                    # -------------------------------------------------
                    # 14. Update progress
                    # -------------------------------------------------

                    processed += len(batch)

                    pbar.update(
                        len(batch)
                    )

            print(
                f"\nProcessed {processed} "
                f"clinical records."
            )

    except Exception as e:

        print(
            f"\nError generating or storing "
            f"embeddings: {e}"
        )

        raise

    finally:

        engine.dispose()

    print(
        "\nPhase 3 complete: clinical records are "
        "vectorized and ready for the hybrid "
        "search pipeline."
    )


if __name__ == "__main__":
    generate_and_store_embeddings()