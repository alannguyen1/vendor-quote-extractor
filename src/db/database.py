"""Database connection and session management.

Provides async SQLite database connectivity using SQLAlchemy 2.0+
with aiosqlite driver. Implements singleton pattern for engine
and async session factory.
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings

# Database file path
DATABASE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "vendor_quote_extractor.db"
)
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Global engine instance (created on first use)
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine.

    Uses sqlite+aiosqlite driver for async SQLite support.
    Engine is created once and reused (singleton pattern).

    Returns:
        AsyncEngine: SQLAlchemy async engine instance.
    """
    global _engine
    if _engine is None:
        # Ensure data directory exists
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(
            DATABASE_URL,
            echo=get_settings().log_level == "DEBUG",
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.

    Returns:
        async_sessionmaker: Factory for creating AsyncSession instances.
    """
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session.

    Yields an async session that auto-closes after use.
    Use with FastAPI's Depends() for route injection.

    Yields:
        AsyncSession: Database session for the request.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database tables.

    Creates all tables defined in models.py if they don't exist.
    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    from src.db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections.

    Should be called on application shutdown to clean up resources.
    """
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
