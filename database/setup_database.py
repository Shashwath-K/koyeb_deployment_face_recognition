import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os

def setup_database():
    base_params = {
        "host": "ep-solitary-shadow-a1pohjxz.ap-southeast-1.pg.koyeb.app",
        "user": "admin",
        "password": "npg_48RFqybgkTad",
        "port": 5432
    }
    
    db_name = "attendance_db"
    sql_file = r"database/schema.sql"
    
    conn = None
    try:
        # Step 1: Create Database
        print(f"Connecting to default 'postgres' database to create {db_name}...")
        conn = psycopg2.connect(database="postgres", **base_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()
        if not exists:
            print(f"Creating database {db_name}...")
            cur.execute(f"CREATE DATABASE {db_name}")
        else:
            print(f"Database {db_name} already exists.")
        
        cur.close()
        conn.close()
        
        # Step 2: Run Schema
        print(f"Connecting to {db_name} to run schema...")
        conn = psycopg2.connect(database=db_name, **base_params)
        conn.autocommit = True
        cur = conn.cursor()
        
        print(f"Reading {sql_file}...")
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql = f.read()
            
        print("Executing SQL commands...")
        cur.execute(sql)
        print("Schema applied successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_database()
