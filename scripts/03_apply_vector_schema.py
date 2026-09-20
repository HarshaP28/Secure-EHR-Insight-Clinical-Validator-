import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path


def apply_pgvector():
    # Find the project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Load environment variables from data/.env
    env_path = project_root / "data" / ".env"
    load_dotenv(env_path, override=True)

    # Read database configuration from .env
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError(
            "Database configuration is incomplete. "
            "Check DB_USER, DB_PASSWORD, DB_HOST, DB_PORT and DB_NAME in data/.env"
        )

    # Build PostgreSQL connection URL
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

    print("Connecting to AWS to apply pgvector schema upgrade...")

    try:
        with engine.begin() as conn:

            # Note:
            # pgvector extension should already be activated by the DBA/superuser.
            # Here we only modify the application table.

            print("Adding 'clinical_embedding' column (768 dimensions)...")

            conn.execute(
                text(
                    """
                    ALTER TABLE patient_encounters
                    ADD COLUMN IF NOT EXISTS clinical_embedding vector(768);
                    """
                )
            )

            # Verify that the column was added
            verify = conn.execute(
                text(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'patient_encounters'
                      AND column_name = 'clinical_embedding';
                    """
                )
            ).fetchone()

            if verify:
                print(
                    f"Schema upgrade complete. "
                    f"Confirmed column: {verify[0]}"
                )
            else:
                print("Column not found after ALTER attempt.")

    except Exception as e:
        print(f"Error applying schema: {e}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    apply_pgvector()