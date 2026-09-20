import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path


def verify_ingestion():

    # ---------------------------------------------------------
    # 1. Find the project root
    # ---------------------------------------------------------

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # ---------------------------------------------------------
    # 2. Load environment variables
    # ---------------------------------------------------------

    env_path = project_root / "data" / ".env"

    load_dotenv(env_path, override=True)

    # ---------------------------------------------------------
    # 3. Read database configuration
    # ---------------------------------------------------------

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # Check that all required values exist
    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError(
            "Database configuration is incomplete. "
            "Please check DB_USER, DB_PASSWORD, DB_HOST, "
            "DB_PORT and DB_NAME in data/.env"
        )

    # ---------------------------------------------------------
    # 4. Create database connection
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

    print("Verifying Data Integrity in AWS PostgreSQL...\n")

    # ---------------------------------------------------------
    # 5. Connect to PostgreSQL
    # ---------------------------------------------------------

    with engine.connect() as conn:

        # -----------------------------------------------------
        # 6. Check total row count
        # -----------------------------------------------------

        count_result = conn.execute(
            text("SELECT COUNT(*) FROM patient_encounters")
        )

        total_records = count_result.scalar()

        print(f"Total records found: {total_records}")

        if total_records == 0:
            print(
                "WARNING: Table is empty. "
                "Ingestion may have failed."
            )
        else:
            print("Data ingestion verified successfully.")

        # -----------------------------------------------------
        # 7. Retrieve sample clinical records
        # -----------------------------------------------------

        print("\nFetching sample clinical records...\n")

        query = text(
            """
            SELECT
                subject_id,
                admission_type,
                drug,
                drg_severity
            FROM patient_encounters
            WHERE drug IS NOT NULL
            LIMIT 10
            """
        )

        # Use pandas for clean terminal formatting
        sample_df = pd.read_sql(query, conn)

        # -----------------------------------------------------
        # 8. Display sample records
        # -----------------------------------------------------

        print("-" * 80)

        if sample_df.empty:
            print("No clinical records with drug information were found.")
        else:
            print(sample_df.to_string(index=False))

        print("-" * 80)

    # ---------------------------------------------------------
    # 9. Close database resources
    # ---------------------------------------------------------

    engine.dispose()

    print("\nVerification complete.")


if __name__ == "__main__":
    verify_ingestion()