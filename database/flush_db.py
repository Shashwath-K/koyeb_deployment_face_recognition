#!/usr/bin/env python3
"""
Flush PostgreSQL database for Koyeb deployment.
This script drops all tables and recreates the database schema.
WARNING: This will delete all data in the database!
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database configuration - update these with your values
DB_CONFIG = {
    "host": "ep-solitary-shadow-a1pohjxz.ap-southeast-1.pg.koyeb.app",
    "user": "admin",
    "password": "npg_48RFqybgkTad",
    "database": "attendance_db",
    "port": 5432
}

def get_connection(use_database=True):
    """Create a database connection."""
    config = DB_CONFIG.copy()
    
    if not use_database:
        # Connect to default 'postgres' database to drop/create databases
        config["database"] = "postgres"
    
    try:
        conn = psycopg2.connect(**config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def flush_database():
    """Flush the database by dropping all tables and sequences."""
    conn = None
    cursor = None
    
    try:
        print(f"Connecting to database: {DB_CONFIG['database']}...")
        conn = get_connection(use_database=True)
        cursor = conn.cursor()
        
        # Disable triggers and drop foreign key constraints first
        print("Disabling triggers and foreign key constraints...")
        
        # Get all tables
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get all sequences
        cursor.execute("""
            SELECT sequence_name 
            FROM information_schema.sequences 
            WHERE sequence_schema = 'public'
        """)
        sequences = [row[0] for row in cursor.fetchall()]
        
        # Drop all tables
        if tables:
            print(f"Dropping {len(tables)} tables...")
            for table in tables:
                try:
                    # Try to drop table with cascade to handle foreign keys
                    drop_query = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(table)
                    )
                    cursor.execute(drop_query)
                    print(f"  Dropped table: {table}")
                except Exception as e:
                    print(f"  Error dropping table {table}: {e}")
        
        # Drop all sequences
        if sequences:
            print(f"Dropping {len(sequences)} sequences...")
            for sequence in sequences:
                try:
                    drop_query = sql.SQL("DROP SEQUENCE IF EXISTS {} CASCADE").format(
                        sql.Identifier(sequence)
                    )
                    cursor.execute(drop_query)
                    print(f"  Dropped sequence: {sequence}")
                except Exception as e:
                    print(f"  Error dropping sequence {sequence}: {e}")
        
        # Drop all views
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public'
        """)
        views = [row[0] for row in cursor.fetchall()]
        
        if views:
            print(f"Dropping {len(views)} views...")
            for view in views:
                try:
                    drop_query = sql.SQL("DROP VIEW IF EXISTS {} CASCADE").format(
                        sql.Identifier(view)
                    )
                    cursor.execute(drop_query)
                    print(f"  Dropped view: {view}")
                except Exception as e:
                    print(f"  Error dropping view {view}: {e}")
        
        # Vacuum to reclaim space
        print("Cleaning up database...")
        cursor.execute("VACUUM")
        
        print("Database flushed successfully!")
        
    except Exception as e:
        print(f"Error flushing database: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def reset_database():
    """Alternative: Drop and recreate the entire database."""
    print("WARNING: This will DROP and RECREATE the entire database!")
    print("All data will be permanently lost!")
    
    confirm = input("Are you sure you want to continue? (yes/NO): ").strip().lower()
    if confirm != 'yes':
        print("Operation cancelled.")
        return
    
    conn = None
    cursor = None
    
    try:
        # Connect to default 'postgres' database
        print("Connecting to PostgreSQL server...")
        conn = get_connection(use_database=False)
        cursor = conn.cursor()
        
        # Terminate all connections to the target database
        print(f"Terminating connections to database '{DB_CONFIG['database']}'...")
        cursor.execute("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = %s AND pid <> pg_backend_pid()
        """, (DB_CONFIG['database'],))
        
        # Drop the database
        print(f"Dropping database '{DB_CONFIG['database']}'...")
        cursor.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(
                sql.Identifier(DB_CONFIG['database'])
            )
        )
        
        # Create a new database
        print(f"Creating database '{DB_CONFIG['database']}'...")
        cursor.execute(
            sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_CONFIG['database'])
            )
        )
        
        print("Database reset successfully!")
        
    except Exception as e:
        print(f"Error resetting database: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def check_connection():
    """Check if database connection works."""
    conn = None
    cursor = None
    
    try:
        print("Testing database connection...")
        conn = get_connection(use_database=True)
        cursor = conn.cursor()
        
        # Get database info
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        
        cursor.execute("SELECT current_database()")
        db_name = cursor.fetchone()[0]
        
        cursor.execute("SELECT current_user")
        user = cursor.fetchone()[0]
        
        print(f"Connected successfully!")
        print(f"PostgreSQL Version: {version}")
        print(f"Database: {db_name}")
        print(f"User: {user}")
        
        # Count tables
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()[0]
        
        print(f"Tables in database: {table_count}")
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    """Main function."""
    print("=" * 60)
    print("PostgreSQL Database Flush Utility")
    print("=" * 60)
    print(f"Host: {DB_CONFIG['host']}")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"User: {DB_CONFIG['user']}")
    print()
    
    # First, check connection
    if not check_connection():
        print("Cannot connect to database. Please check your credentials.")
        return
    
    print()
    print("Select an option:")
    print("1. Flush database (drop all tables)")
    print("2. Reset database (drop and recreate entire database) - MOST DESTRUCTIVE")
    print("3. Check connection only")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\nWARNING: This will delete all data in all tables!")
            confirm = input("Are you sure? (yes/NO): ").strip().lower()
            if confirm == 'yes':
                flush_database()
            else:
                print("Operation cancelled.")
        elif choice == '2':
            reset_database()
        elif choice == '3':
            check_connection()
        elif choice == '4':
            print("Exiting...")
        else:
            print("Invalid choice.")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()