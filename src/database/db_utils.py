import os
import sys
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "oil_maintenance"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASSWORD", "oilwell123"),
    )


def init_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        sql = f.read()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema initialized successfully.")
    finally:
        conn.close()


def execute_query(sql, params=None, fetch=True):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch:
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return cols, rows
            conn.commit()
            return None, None
    finally:
        conn.close()


def bulk_insert(table, columns, rows, page_size=1000):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            col_str = ", ".join(columns)
            sql = f"INSERT INTO {table} ({col_str}) VALUES %s ON CONFLICT DO NOTHING"
            execute_values(cur, sql, rows, page_size=page_size)
        conn.commit()
        print(f"Inserted {len(rows)} rows into {table}.")
    finally:
        conn.close()


if __name__ == "__main__":
    if "--init" in sys.argv:
        init_schema()
