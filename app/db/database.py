from .database_factory import create_database_factory, DatabaseFactory
import os

_db_factory: DatabaseFactory = None


def get_database_factory() -> DatabaseFactory:
    global _db_factory
    if _db_factory is None:
        _db_factory = create_database_factory()
    return _db_factory


async def get_db():
    factory = get_database_factory()
    async for session in factory.get_session():
        yield session


async def init_db():
    factory = get_database_factory()
    await factory.init_db()
