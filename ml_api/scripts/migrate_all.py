"""
Master migration script
Only initializes ML-specific tables (product_embeddings, model_metadata).
Product and rating data is managed by the .NET backend.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_api.scripts.init_database import main as init_database


def main():
    """Run ML table initialization"""
    print("=" * 60)
    print("ML API Migration Script")
    print("Initializes ML-specific tables only")
    print("=" * 60)

    try:
        init_database()
    except Exception as e:
        print(f"\nMigration failed: {e}")
        sys.exit(1)

    print("\nNext steps:")
    print("  1. Ensure .NET backend has populated Products and UserInteraction tables")
    print("  2. Run retrain_models.py to train ML models from the data")
    print("  3. Start the FastAPI server")
    print("=" * 60)


if __name__ == "__main__":
    main()
