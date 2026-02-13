"""
Initialize ML-specific database tables.
Only creates tables owned by the ML API (product_embeddings, model_metadata).
Does NOT touch .NET-managed tables.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_api.database import init_db
from ml_api.models.db_models import ProductEmbedding, ModelMetadata
from ml_api import config


def main():
    """Create ML-specific tables in the database"""
    print("=" * 60)
    print("ML API Database Initialization")
    print("Creates only ML-owned tables (not .NET-managed tables)")
    print("=" * 60)
    print(f"\nDatabase URL: {config.DATABASE_URL[:50]}...")

    try:
        init_db()

        print("\nTables created (if not existing):")
        print("  - product_embeddings (ML-owned)")
        print("  - model_metadata (ML-owned)")

        print("\nTables NOT touched (managed by .NET backend):")
        print('  - "Products"')
        print('  - "ProductCategories"')
        print('  - "UserInteraction"')

        print("\n" + "=" * 60)
        print("Database initialization complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure PostgreSQL is running")
        print("  2. Check DATABASE_URL environment variable")
        print("  3. Ensure the database exists")
        sys.exit(1)


if __name__ == "__main__":
    main()
