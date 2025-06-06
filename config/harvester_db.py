import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Harvester DB config
HARVESTER_DB_USER = os.getenv('DB_USER')
HARVESTER_DB_PASSWORD = os.getenv('DB_PASSWORD')
HARVESTER_DB_HOST = os.getenv('DB_HOST')
HARVESTER_DB_PORT = os.getenv('DB_PORT', '3306')
HARVESTER_DB_NAME = os.getenv('HARVESTER_DB_NAME')

HARVESTER_DATABASE_URL = f"mysql+pymysql://{HARVESTER_DB_USER}:{HARVESTER_DB_PASSWORD}@{HARVESTER_DB_HOST}:{HARVESTER_DB_PORT}/{HARVESTER_DB_NAME}"
print(HARVESTER_DATABASE_URL)
harvester_engine = create_engine(
    HARVESTER_DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True
)
HarvesterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=harvester_engine)

def get_harvester_db():
    db = HarvesterSessionLocal()
    try:
        yield db
    finally:
        db.close()