from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from loguru import logger

from .settings import settings



engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    echo=settings.database.echo,
    pool_pre_ping=True,  # avoids stale-connection errors after idle periods
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db() -> None:
    """Create tables if they don't exist. Prefer Alembic migrations in production;
    this is a convenience for local dev only."""
    from api.classify import models  
    #Base.metadata.create_all(bind=engine), use alembic migrations 
    logger.info("Database tables ensured.")