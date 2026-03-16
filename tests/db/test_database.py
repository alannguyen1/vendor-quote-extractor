"""
Tests for database initialization and connection management (src/db/database.py).

Covers:
- Engine singleton behavior
- Session factory singleton behavior
- Session lifecycle (commit/rollback)
- Database initialization (table creation)
- Connection cleanup on close
- Global state management

These tests complement test_db.py which covers ORM models.
The focus here is on the database.py module functions.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.db.models import Extraction

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path for isolated testing."""
    return tmp_path / "data" / "test_vendor_quote_extractor.db"


@pytest.fixture
def reset_database_globals():
    """Reset database module globals before and after each test.

    This ensures tests don't affect each other through global state.
    """
    import src.db.database as db_module

    # Store original values
    original_engine = db_module._engine
    original_factory = db_module._async_session_factory

    # Reset before test
    db_module._engine = None
    db_module._async_session_factory = None

    yield db_module

    # Cleanup: dispose any engine created during test
    if db_module._engine is not None:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't await in running loop, schedule cleanup
                pass
            else:
                loop.run_until_complete(db_module._engine.dispose())
        except RuntimeError:
            pass

    # Restore original values
    db_module._engine = original_engine
    db_module._async_session_factory = original_factory


@pytest.fixture
async def isolated_db_session(temp_db_path: Path, reset_database_globals):
    """Create an isolated database session using a temp database.

    Patches DATABASE_PATH and DATABASE_URL to use temporary location.
    """
    db_module = reset_database_globals

    # Patch database path to use temp location
    with patch.object(db_module, "DATABASE_PATH", temp_db_path):
        with patch.object(
            db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
        ):
            # Initialize
            await db_module.init_db()

            # Get session
            async for session in db_module.get_session():
                yield session
                break

            # Cleanup
            await db_module.close_db()


# ==============================================================================
# get_engine() Tests
# ==============================================================================


class TestGetEngine:
    """Tests for get_engine() function."""

    def test_returns_async_engine(self, temp_db_path: Path, reset_database_globals):
        """get_engine() should return an AsyncEngine instance."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                engine = db_module.get_engine()

                assert isinstance(engine, AsyncEngine)

    def test_singleton_behavior(self, temp_db_path: Path, reset_database_globals):
        """get_engine() should return the same engine instance on multiple calls."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                engine1 = db_module.get_engine()
                engine2 = db_module.get_engine()

                assert engine1 is engine2

    def test_creates_data_directory(self, temp_db_path: Path, reset_database_globals):
        """get_engine() should create the data directory if it doesn't exist."""
        db_module = reset_database_globals

        # Ensure parent directory doesn't exist
        assert not temp_db_path.parent.exists()

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                db_module.get_engine()

                # Directory should now exist
                assert temp_db_path.parent.exists()

    def test_engine_url_matches_database_url(
        self, temp_db_path: Path, reset_database_globals
    ):
        """Engine should be configured with the correct database URL."""
        db_module = reset_database_globals
        expected_url = f"sqlite+aiosqlite:///{temp_db_path}"

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(db_module, "DATABASE_URL", expected_url):
                engine = db_module.get_engine()

                assert str(engine.url) == expected_url


# ==============================================================================
# get_session_factory() Tests
# ==============================================================================


class TestGetSessionFactory:
    """Tests for get_session_factory() function."""

    def test_returns_async_sessionmaker(
        self, temp_db_path: Path, reset_database_globals
    ):
        """get_session_factory() should return an async_sessionmaker instance."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                factory = db_module.get_session_factory()

                assert isinstance(factory, async_sessionmaker)

    def test_singleton_behavior(self, temp_db_path: Path, reset_database_globals):
        """get_session_factory() should return the same factory on multiple calls."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                factory1 = db_module.get_session_factory()
                factory2 = db_module.get_session_factory()

                assert factory1 is factory2

    def test_factory_bound_to_engine(self, temp_db_path: Path, reset_database_globals):
        """Session factory should be bound to the engine."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                engine = db_module.get_engine()
                factory = db_module.get_session_factory()

                # The factory's bind should be the engine
                assert factory.kw.get("bind") is engine


# ==============================================================================
# get_session() Tests
# ==============================================================================


class TestGetSession:
    """Tests for get_session() async generator."""

    async def test_yields_async_session(
        self, temp_db_path: Path, reset_database_globals
    ):
        """get_session() should yield an AsyncSession instance."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                async for session in db_module.get_session():
                    assert isinstance(session, AsyncSession)
                    # No break - let generator complete to trigger auto-commit

                await db_module.close_db()

    async def test_auto_commits_on_success(
        self, temp_db_path: Path, reset_database_globals
    ):
        """get_session() should auto-commit when no exception occurs."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                # Insert data using get_session
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="test.pdf",
                        file_hash="commit_test_hash",
                        extracted_json="{}",
                    )
                    session.add(extraction)
                    # Don't explicitly commit - should auto-commit when loop completes

                # Verify data persisted (use new session)
                async for session in db_module.get_session():
                    from sqlalchemy import select

                    query = select(Extraction).where(
                        Extraction.file_hash == "commit_test_hash"
                    )
                    result = await session.execute(query)
                    found = result.scalar_one_or_none()
                    assert found is not None
                    assert found.filename == "test.pdf"

                await db_module.close_db()

    async def test_auto_rollback_on_exception(
        self, temp_db_path: Path, reset_database_globals
    ):
        """get_session() should auto-rollback when exception occurs."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                # Insert data then raise exception
                with pytest.raises(ValueError, match="Test exception"):
                    async for session in db_module.get_session():
                        extraction = Extraction(
                            filename="rollback_test.pdf",
                            file_hash="rollback_test_hash",
                            extracted_json="{}",
                        )
                        session.add(extraction)
                        raise ValueError("Test exception")

                # Verify data was NOT persisted (rolled back)
                async for session in db_module.get_session():
                    from sqlalchemy import select

                    query = select(Extraction).where(
                        Extraction.file_hash == "rollback_test_hash"
                    )
                    result = await session.execute(query)
                    found = result.scalar_one_or_none()
                    assert found is None  # Should not exist due to rollback

                await db_module.close_db()


# ==============================================================================
# init_db() Tests
# ==============================================================================


class TestInitDb:
    """Tests for init_db() function."""

    async def test_creates_tables(self, temp_db_path: Path, reset_database_globals):
        """init_db() should create all tables defined in models."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                # Verify tables exist by querying sqlite_master
                engine = db_module.get_engine()
                async with engine.connect() as conn:
                    result = await conn.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    )
                    tables = {row[0] for row in result.fetchall()}

                assert "extractions" in tables
                assert "corrections" in tables

                await db_module.close_db()

    async def test_safe_to_call_multiple_times(
        self, temp_db_path: Path, reset_database_globals
    ):
        """init_db() should be safe to call multiple times (idempotent)."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                # Call init_db multiple times - should not raise
                await db_module.init_db()
                await db_module.init_db()
                await db_module.init_db()

                # Tables should still exist and work
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="idempotent_test.pdf",
                        file_hash="idempotent_hash",
                        extracted_json="{}",
                    )
                    session.add(extraction)

                await db_module.close_db()

    async def test_preserves_existing_data(
        self, temp_db_path: Path, reset_database_globals
    ):
        """init_db() should not delete existing data when called again."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                # Initialize and insert data
                await db_module.init_db()
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="preserve_test.pdf",
                        file_hash="preserve_hash",
                        extracted_json="{}",
                    )
                    session.add(extraction)

                # Call init_db again
                await db_module.init_db()

                # Verify data still exists
                async for session in db_module.get_session():
                    from sqlalchemy import select

                    query = select(Extraction).where(
                        Extraction.file_hash == "preserve_hash"
                    )
                    result = await session.execute(query)
                    found = result.scalar_one_or_none()
                    assert found is not None
                    assert found.filename == "preserve_test.pdf"

                await db_module.close_db()


# ==============================================================================
# close_db() Tests
# ==============================================================================


class TestCloseDb:
    """Tests for close_db() function."""

    async def test_disposes_engine(self, temp_db_path: Path, reset_database_globals):
        """close_db() should dispose the engine."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()
                _engine = db_module.get_engine()

                # Engine should be active
                assert db_module._engine is not None

                await db_module.close_db()

                # Engine reference should be cleared
                assert db_module._engine is None

    async def test_clears_session_factory(
        self, temp_db_path: Path, reset_database_globals
    ):
        """close_db() should clear the session factory reference."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()
                db_module.get_session_factory()  # Create factory

                # Factory should exist
                assert db_module._async_session_factory is not None

                await db_module.close_db()

                # Factory reference should be cleared
                assert db_module._async_session_factory is None

    async def test_allows_reinit_after_close(
        self, temp_db_path: Path, reset_database_globals
    ):
        """After close_db(), calling init_db() should work again."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                # First init/close cycle
                await db_module.init_db()
                await db_module.close_db()

                # Second init should work
                await db_module.init_db()

                # Should be able to use database
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="reinit_test.pdf",
                        file_hash="reinit_hash",
                        extracted_json="{}",
                    )
                    session.add(extraction)

                await db_module.close_db()

    async def test_safe_to_call_when_not_initialized(self, reset_database_globals):
        """close_db() should be safe to call even if engine was never created."""
        db_module = reset_database_globals

        # Engine is None, close_db should not raise
        assert db_module._engine is None
        await db_module.close_db()  # Should not raise


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestDatabaseIntegration:
    """Integration tests for the full database lifecycle."""

    async def test_full_lifecycle(self, temp_db_path: Path, reset_database_globals):
        """Test complete init -> use -> close cycle."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                # 1. Initialize
                await db_module.init_db()

                # 2. Insert data
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="lifecycle_test.pdf",
                        file_hash="lifecycle_hash",
                        extracted_json='{"test": true}',
                        overall_confidence=0.95,
                        requires_review=False,
                    )
                    session.add(extraction)

                # 3. Query data
                async for session in db_module.get_session():
                    from sqlalchemy import select

                    query = select(Extraction).where(
                        Extraction.filename == "lifecycle_test.pdf"
                    )
                    result = await session.execute(query)
                    found = result.scalar_one()
                    assert found.file_hash == "lifecycle_hash"
                    assert found.overall_confidence == 0.95

                # 4. Close
                await db_module.close_db()

                # 5. Verify database file exists on disk
                assert temp_db_path.exists()

    async def test_concurrent_sessions(
        self, temp_db_path: Path, reset_database_globals
    ):
        """Test that multiple sessions can operate concurrently."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                # Create multiple extractions concurrently
                import asyncio

                async def create_extraction(hash_suffix: str):
                    async for session in db_module.get_session():
                        extraction = Extraction(
                            filename=f"concurrent_{hash_suffix}.pdf",
                            file_hash=f"concurrent_hash_{hash_suffix}",
                            extracted_json="{}",
                        )
                        session.add(extraction)

                # Run 5 concurrent insertions
                await asyncio.gather(
                    create_extraction("1"),
                    create_extraction("2"),
                    create_extraction("3"),
                    create_extraction("4"),
                    create_extraction("5"),
                )

                # Verify all were created
                async for session in db_module.get_session():
                    from sqlalchemy import func, select

                    result = await session.execute(
                        select(func.count()).select_from(Extraction)
                    )
                    count = result.scalar()
                    assert count == 5

                await db_module.close_db()


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Edge case tests for database module."""

    async def test_empty_database_query(
        self, temp_db_path: Path, reset_database_globals
    ):
        """Querying empty database should return no results, not error."""
        db_module = reset_database_globals

        with patch.object(db_module, "DATABASE_PATH", temp_db_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{temp_db_path}"
            ):
                await db_module.init_db()

                async for session in db_module.get_session():
                    from sqlalchemy import select

                    result = await session.execute(select(Extraction))
                    all_extractions = result.scalars().all()
                    assert all_extractions == []

                await db_module.close_db()

    async def test_database_path_with_special_characters(
        self, tmp_path: Path, reset_database_globals
    ):
        """Database path with spaces should work correctly."""
        db_module = reset_database_globals

        special_path = tmp_path / "data with spaces" / "test db.db"

        with patch.object(db_module, "DATABASE_PATH", special_path):
            with patch.object(
                db_module, "DATABASE_URL", f"sqlite+aiosqlite:///{special_path}"
            ):
                await db_module.init_db()

                # Should work normally
                async for session in db_module.get_session():
                    extraction = Extraction(
                        filename="special_path_test.pdf",
                        file_hash="special_path_hash",
                        extracted_json="{}",
                    )
                    session.add(extraction)

                await db_module.close_db()

                # File should exist
                assert special_path.exists()
