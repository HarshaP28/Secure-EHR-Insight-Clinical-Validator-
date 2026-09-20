import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path


# Environment variables:
# Values such as database username, password, host, port and database name
# are stored outside the Python code in the .env file.


# Data ingestion:
# Taking data from a source such as a CSV file and bringing it
# into a database so that the data can be stored and used.


def ingest_data():

    # ---------------------------------------------------------
    # 1. Find the project root dynamically
    # ---------------------------------------------------------

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # ---------------------------------------------------------
    # 2. Load environment variables
    # ---------------------------------------------------------

    env_path = project_root / "data" / ".env"
    load_dotenv(env_path, override=True)

    # ---------------------------------------------------------
    # 3. Build paths to CSV and schema
    # ---------------------------------------------------------

    csv_path = project_root / "data" / "MIMIC_IV_Transcrpt.csv"
    schema_path = project_root / "src" / "database" / "schema.sql"

    # ---------------------------------------------------------
    # 4. Check that required files exist
    # ---------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Could not find dataset at {csv_path}"
        )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Could not find schema at {schema_path}"
        )

    # ---------------------------------------------------------
    # 5. Read database configuration from .env
    # ---------------------------------------------------------

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # Check that required environment variables exist
    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError(
            "Database configuration is incomplete. "
            "Check DB_USER, DB_PASSWORD, DB_HOST, DB_PORT and DB_NAME in data/.env"
        )

    # ---------------------------------------------------------
    # 6. Create PostgreSQL database connection
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
    # 7. Create database table using schema.sql
    # ---------------------------------------------------------

    print(f"Executing schema setup from:\n {schema_path}")

    with engine.begin() as conn:
        with open(schema_path, "r", encoding="utf-8") as file:
            schema_sql = file.read()

        conn.execute(text(schema_sql))

    print("Schema setup complete.")

    # ---------------------------------------------------------
    # 8. Load CSV data
    # ---------------------------------------------------------

    print(f"Loading clinical data from:\n  {csv_path}")

    # low_memory=False prevents the mixed-type DtypeWarning
    # while pandas reads the CSV.
    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    # ---------------------------------------------------------
    # 9. Replace missing values with None
    # ---------------------------------------------------------

    df = df.where(pd.notnull(df), None)

    print(f"Total records loaded: {len(df)}")
    print("Starting data ingestion into AWS PostgreSQL...")

    # ---------------------------------------------------------
    # 10. Insert data in smaller batches
    # ---------------------------------------------------------

    chunk_size = 1000

    total_records = len(df)

    for start in range(0, total_records, chunk_size):

        end = min(start + chunk_size, total_records)

        chunk = df.iloc[start:end]

        # Each chunk gets its own database transaction.
        # If one chunk fails, it does not invalidate the entire
        # 232,158-record transaction.
        with engine.begin() as conn:

            chunk.to_sql(
                "patient_encounters",
                conn,
                if_exists="append",
                index=False,
                method="multi"
            )

        print(
            f"Inserted records {start + 1} - {end} "
            f"of {total_records}"
        )

    # ---------------------------------------------------------
    # 11. Finish
    # ---------------------------------------------------------

    print("Baseline legacy data ingestion complete.")

    engine.dispose()


if __name__ == "__main__":
    ingest_data()