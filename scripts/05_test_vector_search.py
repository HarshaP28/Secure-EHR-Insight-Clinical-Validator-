import os
from pathlib import Path

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def test_vector_search():

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
    # 4. Load local BioClinical ModernBERT model
    # ---------------------------------------------------------

    print("Loading local BioClinical ModernBERT model...")

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
    # 6. Define a complex, natural language medical query
    # ---------------------------------------------------------

    query_text = (
        "Patient presenting with severe liver disease "
        "and fluid retention needing diuretics"
    )

    print(
        f"\nSemantic Query: '{query_text}'"
    )

    # ---------------------------------------------------------
    # 7. Vectorize the query locally
    # ---------------------------------------------------------

    query_vector = model.encode(
        query_text
    ).tolist()

    print(
        f"Query vector generated with "
        f"{len(query_vector)} dimensions."
    )

    # ---------------------------------------------------------
    # 8. Search PostgreSQL using pgvector
    #    cosine distance operator (<=>)
    # ---------------------------------------------------------

    search_sql = text(
        """
        SELECT
            id,
            description,
            drug,
            clinical_embedding <=> CAST(
                :query_vector AS vector
            ) AS cosine_distance
        FROM patient_encounters
        WHERE clinical_embedding IS NOT NULL
        ORDER BY cosine_distance ASC
        LIMIT 3;
        """
    )

    # ---------------------------------------------------------
    # 9. Execute semantic search
    # ---------------------------------------------------------

    try:

        with engine.connect() as conn:

            results = conn.execute(
                search_sql,
                {
                    "query_vector": str(query_vector)
                }
            ).mappings().fetchall()

            print("\nTop 3 Semantic Matches:")

            if not results:
                print(
                    "No records with clinical embeddings "
                    "were found."
                )
                return

            for rank, row in enumerate(results, 1):

                print("-" * 65)

                print(
                    f"Rank {rank} "
                    f"(Distance: "
                    f"{row['cosine_distance']:.4f})"
                )

                print(
                    f"Diagnosis : {row['description']}"
                )

                print(
                    f"Drug      : {row['drug']}"
                )

                print(
                    f"Record ID : {row['id']}"
                )

    except Exception as e:

        print(
            f"\nError during vector search: {e}"
        )

        raise

    finally:

        engine.dispose()


if __name__ == "__main__":
    test_vector_search()