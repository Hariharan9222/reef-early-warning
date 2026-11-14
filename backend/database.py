"""
backend/database.py
SQLAlchemy database setup supporting PostgreSQL+PostGIS with local SQLite fallback.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

# SQLite requires specific connect_args for multithreading
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

# Create directory for SQLite if needed
if "sqlite" in settings.DATABASE_URL:
    db_file = settings.DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

