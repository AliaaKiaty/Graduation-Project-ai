# Database Migration Scripts

This directory contains scripts for migrating data from CSV files to PostgreSQL.

## Prerequisites

1. **PostgreSQL installed and running**
   ```bash
   # Windows: Download from https://www.postgresql.org/download/windows/
   # Linux: sudo apt-get install postgresql
   # macOS: brew install postgresql
   ```

2. **Create database**
   ```bash
   createdb ml_recommendation
   ```

3. **Set environment variable** (optional, defaults to localhost)
   ```bash
   export DATABASE_URL="postgresql://postgres:password@localhost:5432/ml_recommendation"
   ```

4. **Install dependencies**
   ```bash
   cd ml_api
   pip install -r requirements.txt
   ```

5. **Ensure CSV files exist**
   - `content/product_descriptions.csv` (~124K products)
   - `ratings_Beauty.csv` (~2M ratings)

## Scripts Overview

### 1. `init_database.py` - Initialize Database Schema
Creates all database tables (categories, products, ratings, product_embeddings, model_metadata).

**Usage:**
```bash
python -m ml_api.scripts.init_database
```

**Output:**
- Creates 6 tables in PostgreSQL
- Prints list of created tables

### 2. `import_products.py` - Import Products
Imports products from `content/product_descriptions.csv` into the products table.

**Features:**
- Creates 9 default categories (Tools, Paint, Electrical, Plumbing, Hardware, Lumber, Lighting, Outdoor, Uncategorized)
- Automatically assigns categories based on keyword matching
- Extracts product names from descriptions
- Bulk inserts in batches of 1000 for performance

**Usage:**
```bash
python -m ml_api.scripts.import_products
```

**Expected Result:**
- ~124,000 products imported
- Categories created with hierarchical structure

### 3. `import_ratings.py` - Import Ratings
Imports ratings from `ratings_Beauty.csv` into the ratings table.

**Features:**
- Maps product UIDs to internal database IDs
- Skips ratings for products not in database
- Handles duplicate user-product pairs gracefully
- Bulk inserts in batches of 5000 for performance

**Usage:**
```bash
python -m ml_api.scripts.import_ratings
```

**Expected Result:**
- ~2,000,000 ratings imported
- Skips ratings for non-existent products

### 4. `migrate_all.py` - Master Migration (Recommended)
Runs all migration steps in sequence: init → products → ratings

**Usage:**
```bash
python -m ml_api.scripts.migrate_all
```

**This is the recommended way to run the full migration.**

## Migration Steps (Manual)

If you prefer to run scripts individually:

```bash
# Step 1: Initialize database
python -m ml_api.scripts.init_database

# Step 2: Import products
python -m ml_api.scripts.import_products

# Step 3: Import ratings
python -m ml_api.scripts.import_ratings
```

## Verification

After migration, verify data was imported correctly:

```sql
-- Connect to database
psql -d ml_recommendation

-- Check table counts
SELECT COUNT(*) FROM categories;      -- Should be ~10
SELECT COUNT(*) FROM products;        -- Should be ~124,000
SELECT COUNT(*) FROM ratings;         -- Should be ~2,000,000

-- Check top products by rating count
SELECT p.product_uid, p.name, COUNT(r.id) as rating_count
FROM products p
JOIN ratings r ON p.id = r.product_id
GROUP BY p.id, p.product_uid, p.name
ORDER BY rating_count DESC
LIMIT 10;

-- Check category distribution
SELECT c.name, COUNT(p.id) as product_count
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name
ORDER BY product_count DESC;
```

## Troubleshooting

### Error: "Database 'ml_recommendation' does not exist"
```bash
createdb ml_recommendation
```

### Error: "psycopg2 not found"
```bash
pip install psycopg2-binary==2.9.9
```

### Error: "Permission denied"
Ensure your PostgreSQL user has CREATE and INSERT permissions:
```sql
GRANT ALL PRIVILEGES ON DATABASE ml_recommendation TO your_user;
```

### Error: "CSV file not found"
Ensure CSV files are in the correct locations:
- `content/product_descriptions.csv`
- `ratings_Beauty.csv` (project root)

### Migration runs slowly
This is normal for large datasets. Expected times:
- Products import: ~2-5 minutes
- Ratings import: ~10-20 minutes
- Total: ~15-25 minutes

## Next Steps

After successful migration:

1. **Verify data integrity**
   ```bash
   python -m ml_api.scripts.verify_migration  # (to be created)
   ```

2. **Start API server**
   ```bash
   cd ml_api
   uvicorn main:app --reload
   ```

3. **Retrain ML models**
   ```bash
   python -m ml_api.scripts.retrain_models  # (to be created in Phase 6)
   ```

## Notes

- Migration is idempotent for products (uses product_uid as unique key)
- Duplicate ratings are skipped (unique constraint on user_id + product_id)
- Batch inserts optimize performance for large datasets
- Category assignment is automatic based on keywords
- All scripts include error handling and progress reporting
