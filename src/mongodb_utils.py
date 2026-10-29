import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from common import ROOT, fail

load_dotenv(ROOT / ".env")


def get_db():
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    name = os.getenv("MONGO_DATABASE", "financial_analytics")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
    except PyMongoError as e:
        fail(f"MongoDB is not reachable at {uri}\n"
             "Start it (Windows: run 'net start MongoDB' in an Administrator terminal, or use Atlas and set MONGO_URI in .env).\n"
             f"Details: {e}")
    return client[name]


def save_collections(db, docs: dict):
    for name, items in docs.items():
        db[name].delete_many({})
        if items:
            db[name].insert_many(items)


def save_metadata(db, **meta):
    db["run_metadata"].delete_many({})
    db["run_metadata"].insert_one({**meta, "finished_utc": datetime.now(timezone.utc).isoformat()})
