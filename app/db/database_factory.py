from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
import os


class DatabaseFactory(ABC):
    @abstractmethod
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        pass

    @abstractmethod
    async def init_db(self):
        pass

    @abstractmethod
    def get_session_factory(self):
        pass


class SQLiteDatabaseFactory(DatabaseFactory):
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "sqlite+aiosqlite:///./pdf_processing.db"
        )
        self.engine: AsyncEngine = create_async_engine(
            self.database_url,
            echo=False,
            future=True,
            poolclass=NullPool,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    async def init_db(self):
        from app.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def get_session_factory(self):
        return self.session_factory


class PostgreSQLDatabaseFactory(DatabaseFactory):
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: AsyncEngine = create_async_engine(
            self.database_url,
            echo=False,
            future=True,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    async def init_db(self):
        from app.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def get_session_factory(self):
        return self.session_factory


def create_database_factory(database_url: str = None) -> DatabaseFactory:
    database_url = database_url or os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./pdf_processing.db")
    
    if "postgresql" in database_url or "postgres" in database_url:
        return PostgreSQLDatabaseFactory(database_url)
    else:
        return SQLiteDatabaseFactory(database_url)
