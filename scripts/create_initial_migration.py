"""
Create Initial Database Migration

This script creates the initial Alembic migration for all database tables.
Run this after setting up PostgreSQL to generate the schema migration.

Usage:
    uv run python scripts/create_initial_migration.py
"""

import subprocess
import sys
from pathlib import Path

def create_migration():
    """Create initial migration with autogenerate."""
    print("Creating initial database migration...")
    
    # Run alembic revision with autogenerate
    cmd = [
        sys.executable, "-m", "alembic",
        "revision",
        "--autogenerate",
        "-m", "initial_schema"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("\n✓ Initial migration created successfully!")
        print("\nNext steps:")
        print("1. Review the migration file in migrations/versions/")
        print("2. Set up PostgreSQL database")
        print("3. Run migration: alembic upgrade head")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating migration: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    create_migration()
