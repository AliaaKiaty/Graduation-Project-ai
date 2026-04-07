"""
Schema v2 migration script.

Alters existing .NET-backend tables to match the v2 schema from the PDF:
  - ProductCategories: add NameEn, NameAr, CreatedAt, CreatedBy, IsDeleted, UpdatedAt
                       (copies existing Name → NameEn for backward compat)
  - Products:          add NameEn, NameAr, DescriptionEn, DescriptionAr,
                       CreatedAt, CreatedBy, IsDeleted, UpdatedAt, RowVersion
                       (copies existing Name → NameEn, Description → DescriptionEn)
  - UserInteraction:   add CreatedAt, CreatedBy, IsDeleted, UpdatedAt
  - RawMaterialCategories: CREATE TABLE (new)
  - RawMaterials:          CREATE TABLE (new)
  - product_embeddings / model_metadata: created as before (ML-owned)

Safe to run multiple times — each step is wrapped in its own try/except
and uses IF NOT EXISTS / DO NOTHING semantics.

Usage:
  python -m ml_api.scripts.migrate_v2

Or trigger via API:
  POST /admin/migrate-v2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from ml_api.database import engine, init_db


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

def _run(conn, sql: str, description: str) -> str:
    """Execute a single DDL statement and return a status string."""
    try:
        conn.execute(text(sql))
        return f"OK  {description}"
    except Exception as e:
        msg = str(e).split("\n")[0]
        return f"SKIP {description}: {msg}"


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def migrate(verbose: bool = True) -> dict:
    """
    Run all v2 migration steps.

    Returns:
        dict with 'steps' (list of status strings) and 'error' (str or None).
    """
    steps = []
    error = None

    try:
        with engine.begin() as conn:

            # ── ProductCategories ─────────────────────────────────────────
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "NameEn" VARCHAR(255)',
                'ProductCategories.NameEn add'))
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "NameAr" VARCHAR(255)',
                'ProductCategories.NameAr add'))
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "CreatedAt" TIMESTAMP',
                'ProductCategories.CreatedAt add'))
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "CreatedBy" VARCHAR(450)',
                'ProductCategories.CreatedBy add'))
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "IsDeleted" BOOLEAN DEFAULT FALSE',
                'ProductCategories.IsDeleted add'))
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ADD COLUMN IF NOT EXISTS "UpdatedAt" TIMESTAMP',
                'ProductCategories.UpdatedAt add'))
            # Migrate existing Name → NameEn where NameEn is still NULL
            steps.append(_run(conn,
                'UPDATE "ProductCategories" SET "NameEn" = "Name" WHERE "NameEn" IS NULL AND "Name" IS NOT NULL',
                'ProductCategories: copy Name → NameEn'))
            # Make legacy Name column nullable (v2 uses NameEn/NameAr instead)
            steps.append(_run(conn,
                'ALTER TABLE "ProductCategories" ALTER COLUMN "Name" DROP NOT NULL',
                'ProductCategories.Name drop NOT NULL'))

            # ── Products ──────────────────────────────────────────────────
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "NameEn" VARCHAR(500)',
                'Products.NameEn add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "NameAr" VARCHAR(500)',
                'Products.NameAr add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "DescriptionEn" TEXT',
                'Products.DescriptionEn add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "DescriptionAr" TEXT',
                'Products.DescriptionAr add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "CreatedAt" TIMESTAMP',
                'Products.CreatedAt add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "CreatedBy" VARCHAR(450)',
                'Products.CreatedBy add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "IsDeleted" BOOLEAN DEFAULT FALSE',
                'Products.IsDeleted add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "UpdatedAt" TIMESTAMP',
                'Products.UpdatedAt add'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ADD COLUMN IF NOT EXISTS "RowVersion" BYTEA',
                'Products.RowVersion add'))
            # Migrate existing Name → NameEn, Description → DescriptionEn
            steps.append(_run(conn,
                'UPDATE "Products" SET "NameEn" = "Name" WHERE "NameEn" IS NULL AND "Name" IS NOT NULL',
                'Products: copy Name → NameEn'))
            steps.append(_run(conn,
                'UPDATE "Products" SET "DescriptionEn" = "Description" WHERE "DescriptionEn" IS NULL AND "Description" IS NOT NULL',
                'Products: copy Description → DescriptionEn'))
            # Make legacy Name/Description nullable (v2 uses NameEn/NameAr instead)
            steps.append(_run(conn,
                'ALTER TABLE "Products" ALTER COLUMN "Name" DROP NOT NULL',
                'Products.Name drop NOT NULL'))
            steps.append(_run(conn,
                'ALTER TABLE "Products" ALTER COLUMN "Description" DROP NOT NULL',
                'Products.Description drop NOT NULL'))

            # ── UserInteraction ───────────────────────────────────────────
            steps.append(_run(conn,
                'ALTER TABLE "UserInteraction" ADD COLUMN IF NOT EXISTS "CreatedAt" TIMESTAMP',
                'UserInteraction.CreatedAt add'))
            steps.append(_run(conn,
                'ALTER TABLE "UserInteraction" ADD COLUMN IF NOT EXISTS "CreatedBy" VARCHAR(450)',
                'UserInteraction.CreatedBy add'))
            steps.append(_run(conn,
                'ALTER TABLE "UserInteraction" ADD COLUMN IF NOT EXISTS "IsDeleted" BOOLEAN DEFAULT FALSE',
                'UserInteraction.IsDeleted add'))
            steps.append(_run(conn,
                'ALTER TABLE "UserInteraction" ADD COLUMN IF NOT EXISTS "UpdatedAt" TIMESTAMP',
                'UserInteraction.UpdatedAt add'))

            # ── RawMaterialCategories (new table) ─────────────────────────
            steps.append(_run(conn, """
                CREATE TABLE IF NOT EXISTS "RawMaterialCategories" (
                    "Id"        SERIAL PRIMARY KEY,
                    "NameEn"    VARCHAR(255),
                    "NameAr"    VARCHAR(255),
                    "Image"     VARCHAR(1000),
                    "CreatedAt" TIMESTAMP,
                    "CreatedBy" VARCHAR(450),
                    "IsDeleted" BOOLEAN DEFAULT FALSE,
                    "UpdatedAt" TIMESTAMP
                )
            """, 'RawMaterialCategories table create'))

            # ── RawMaterials (new table) ──────────────────────────────────
            steps.append(_run(conn, """
                CREATE TABLE IF NOT EXISTS "RawMaterials" (
                    "Id"            SERIAL PRIMARY KEY,
                    "NameEn"        VARCHAR(500),
                    "NameAr"        VARCHAR(500),
                    "ImageUrl"      VARCHAR(1000),
                    "Quantity"      INTEGER,
                    "Price"         NUMERIC(18, 2),
                    "DescriptionEn" TEXT,
                    "DescriptionAr" TEXT,
                    "SupplierID"    VARCHAR(450),
                    "CategoryId"    INTEGER REFERENCES "RawMaterialCategories"("Id"),
                    "IsDeleted"     BOOLEAN DEFAULT FALSE,
                    "UpdatedAt"     TIMESTAMP,
                    "RowVersion"    BYTEA
                )
            """, 'RawMaterials table create'))

            # Add FK on UserInteraction.RawMaterialID if not already there
            steps.append(_run(conn, """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE constraint_name = 'fk_userinteraction_rawmaterial'
                    ) THEN
                        ALTER TABLE "UserInteraction"
                            ADD CONSTRAINT fk_userinteraction_rawmaterial
                            FOREIGN KEY ("RawMaterialID") REFERENCES "RawMaterials"("Id");
                    END IF;
                END$$
            """, 'UserInteraction.RawMaterialID FK add'))

            # ── product_embeddings FK: point to real lowercase products table ──
            # The table was originally created with FK → "Products"("Id") (PascalCase).
            # We need it to reference products(id) (lowercase, where real data lives).
            steps.append(_run(conn, """
                DO $$
                DECLARE
                    fk_name TEXT;
                BEGIN
                    -- Find existing FK constraint name pointing at "Products"
                    SELECT conname INTO fk_name
                    FROM pg_constraint
                    WHERE conrelid = 'product_embeddings'::regclass
                      AND contype = 'f';

                    IF fk_name IS NOT NULL THEN
                        EXECUTE 'ALTER TABLE product_embeddings DROP CONSTRAINT ' || quote_ident(fk_name);
                    END IF;

                    -- Re-add FK pointing at the real lowercase products table
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'product_embeddings'::regclass
                          AND contype = 'f'
                    ) THEN
                        ALTER TABLE product_embeddings
                            ADD CONSTRAINT product_embeddings_product_id_fkey
                            FOREIGN KEY (product_id)
                            REFERENCES products(id)
                            ON DELETE CASCADE;
                    END IF;
                END$$
            """, 'product_embeddings FK → products(id) fix'))

            # ── ML-owned tables (create_all handles idempotently) ─────────
            init_db()
            steps.append("OK  ML-owned tables (product_embeddings, model_metadata)")

    except Exception as e:
        error = str(e)

    if verbose:
        for s in steps:
            print(s)
        if error:
            print(f"\nERROR: {error}")

    return {"steps": steps, "error": error}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running schema v2 migration...\n")
    result = migrate(verbose=True)
    print("\nDone." if not result["error"] else f"\nFailed: {result['error']}")
    sys.exit(1 if result["error"] else 0)
