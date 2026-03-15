"""
Seed script — inserts test data into the database for all ML endpoints.

Inserts:
  - 8 product categories
  - 40 products (5 per category) with Arabic names + English descriptions
  - 300 user interactions (ratings) from 15 simulated users
  - product_embeddings: cluster IDs predicted by the live TF-IDF + KMeans model

After running, the following endpoints will return real data:
  POST /recommend/popular          — rankings from rating counts
  POST /recommend/content-based    — cluster-matched products
  GET  /recommend/products         — paginated product list
  GET  /recommend/categories       — category list

  NOTE: /recommend/collaborative requires the SVD model to be retrained
        on this data first. Call POST /admin/retrain after seeding.

Usage:
  # Run directly (from the project root)
  python -m ml_api.scripts.seed_data

  # Or call POST /admin/seed from any HTTP client (no body required)
"""

import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_api.database import SessionLocal, init_db
from ml_api.models.db_models import (
    ProductCategory, Product, UserInteraction, ProductEmbedding
)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

CATEGORIES = [
    {"Id": 1,  "NameAr": "العناية بالبشرة",  "NameEn": "Skincare",       "Image": "https://example.com/images/skincare.jpg"},
    {"Id": 2,  "NameAr": "العطور",            "NameEn": "Perfumes",       "Image": "https://example.com/images/perfumes.jpg"},
    {"Id": 3,  "NameAr": "المكياج",           "NameEn": "Makeup",         "Image": "https://example.com/images/makeup.jpg"},
    {"Id": 4,  "NameAr": "العناية بالشعر",    "NameEn": "Hair Care",      "Image": "https://example.com/images/haircare.jpg"},
    {"Id": 5,  "NameAr": "الإلكترونيات",      "NameEn": "Electronics",    "Image": "https://example.com/images/electronics.jpg"},
    {"Id": 6,  "NameAr": "الملابس",           "NameEn": "Clothing",       "Image": "https://example.com/images/clothing.jpg"},
    {"Id": 7,  "NameAr": "المنزل والمطبخ",    "NameEn": "Home & Kitchen", "Image": "https://example.com/images/home.jpg"},
    {"Id": 8,  "NameAr": "الرياضة",           "NameEn": "Sports",         "Image": "https://example.com/images/sports.jpg"},
]

# Products: Arabic Name, English description (for TF-IDF), category_id, price
PRODUCTS = [
    # ── Skincare (cat 1) ────────────────────────────────────────────────────
    (1,  "كريم الترطيب اليومي",
     "Daily moisturizing face cream with hyaluronic acid and vitamin E for deep hydration and skin barrier repair.",
     1, 89.99),
    (2,  "سيروم فيتامين سي",
     "Brightening vitamin C serum with 15% ascorbic acid to reduce dark spots and even skin tone.",
     1, 149.00),
    (3,  "غسول الوجه المنقي",
     "Gentle purifying face cleanser with salicylic acid for oily and acne-prone skin deep cleansing.",
     1, 65.50),
    (4,  "واقي الشمس SPF50",
     "Lightweight sunscreen SPF50 with broad spectrum UVA UVB protection and non-greasy formula.",
     1, 110.00),
    (5,  "كريم العيون المضاد للتجاعيد",
     "Anti-aging eye cream with retinol and peptides to reduce dark circles puffiness and fine lines.",
     1, 195.00),

    # ── Perfumes (cat 2) ────────────────────────────────────────────────────
    (6,  "عطر عود الملكي",
     "Luxurious royal oud perfume with rich woody and amber notes for long lasting oriental fragrance.",
     2, 450.00),
    (7,  "عطر الورد الطائفي",
     "Fresh floral rose perfume inspired by Taif roses with delicate jasmine and musk base notes.",
     2, 320.00),
    (8,  "عطر المسك الأبيض",
     "Clean white musk perfume with soft powdery notes ideal for everyday casual wear.",
     2, 210.00),
    (9,  "عطر البخور الفاخر",
     "Traditional Arabic bakhoor incense perfume with sandalwood and resin for a bold statement.",
     2, 380.00),
    (10, "عطر الزهور الربيعي",
     "Light spring floral perfume with citrus top notes and peony heart for feminine elegance.",
     2, 275.00),

    # ── Makeup (cat 3) ──────────────────────────────────────────────────────
    (11, "أحمر الشفاه المخملي",
     "Long-lasting matte velvet lipstick with full coverage formula available in 20 bold shades.",
     3, 75.00),
    (12, "فاونديشن السيولة الكاملة",
     "Full coverage liquid foundation with 24 hour wear buildable formula for flawless skin finish.",
     3, 135.00),
    (13, "ماسكارا مكثفة",
     "Volumizing and lengthening mascara with waterproof formula for dramatic lash definition.",
     3, 89.00),
    (14, "بودرة التثبيت الشفافة",
     "Translucent setting powder to control shine and lock makeup in place for all day wear.",
     3, 99.00),
    (15, "بلاش الخدود المضيء",
     "Silky blush powder with pearl shimmer for natural rosy flush and highlighted cheekbones.",
     3, 85.00),

    # ── Hair Care (cat 4) ───────────────────────────────────────────────────
    (16, "شامبو تغذية مكثفة",
     "Intensive nourishing shampoo with argan oil and keratin protein for damaged and dry hair repair.",
     4, 95.00),
    (17, "بلسم ترطيب عميق",
     "Deep moisturizing conditioner with coconut milk and shea butter for soft smooth manageable hair.",
     4, 88.00),
    (18, "ماسك الشعر بالكيراتين",
     "Professional keratin hair mask treatment to eliminate frizz and add shine and smoothness.",
     4, 145.00),
    (19, "زيت الأرغان للشعر",
     "Pure Moroccan argan oil hair serum to protect against heat and add glossy shine to all hair types.",
     4, 120.00),
    (20, "سيروم تساقط الشعر",
     "Anti hair loss serum with biotin and caffeine to stimulate hair follicles and promote growth.",
     4, 175.00),

    # ── Electronics (cat 5) ─────────────────────────────────────────────────
    (21, "سماعات لاسلكية بلوتوث",
     "Premium wireless Bluetooth headphones with active noise cancellation and 30 hour battery life.",
     5, 899.00),
    (22, "شاحن لاسلكي سريع",
     "Fast wireless charger 15W compatible with all Qi enabled smartphones and earbuds cases.",
     5, 145.00),
    (23, "ساعة ذكية رياضية",
     "Smart fitness watch with heart rate monitor GPS tracking sleep analysis and 7 day battery.",
     5, 1250.00),
    (24, "مكبر صوت بلوتوث مقاوم للماء",
     "Waterproof portable Bluetooth speaker with 360 degree sound and 12 hour playtime.",
     5, 520.00),
    (25, "قلم لمس ذكي",
     "Precision stylus pen for tablets and smartphones with tilt sensitivity and pressure control.",
     5, 189.00),

    # ── Clothing (cat 6) ────────────────────────────────────────────────────
    (26, "عباية تطريز فاخر",
     "Luxury embroidered abaya with intricate floral embroidery on premium crepe fabric.",
     6, 650.00),
    (27, "فستان سهرة أنيق",
     "Elegant evening dress with chiffon overlay and beaded waist belt for formal occasions.",
     6, 890.00),
    (28, "بلوزة قطن كاجوال",
     "Casual cotton blend blouse with relaxed fit and breathable fabric for everyday comfort.",
     6, 145.00),
    (29, "بنطلون جينز كلاسيكي",
     "Classic slim fit denim jeans with stretch fabric and five pocket design for modern style.",
     6, 280.00),
    (30, "وشاح حرير طبيعي",
     "Natural silk scarf with hand rolled edges and traditional geometric pattern in vibrant colors.",
     6, 320.00),

    # ── Home & Kitchen (cat 7) ──────────────────────────────────────────────
    (31, "طقم أواني طبخ غرانيت",
     "Non-stick granite coated cookware set 10 pieces with tempered glass lids and ergonomic handles.",
     7, 750.00),
    (32, "مصفاة قهوة فرنسية",
     "Stainless steel French press coffee maker 1 liter with double filter for rich full bodied brew.",
     7, 185.00),
    (33, "مجموعة سكاكين المطبخ",
     "Professional kitchen knife set 6 pieces with German stainless steel blades and pakka wood handles.",
     7, 420.00),
    (34, "خلاط كهربائي متعدد الوظائف",
     "Multifunctional electric blender 1500W with smoothie maker chopper and food processor attachments.",
     7, 650.00),
    (35, "مفرش طاولة قطن مطرز",
     "Hand embroidered cotton table runner with traditional Arabic geometric design for elegant table setting.",
     7, 95.00),

    # ── Sports (cat 8) ──────────────────────────────────────────────────────
    (36, "حصيرة يوغا مضادة للانزلاق",
     "Non-slip eco friendly yoga mat 6mm thick with alignment lines and carrying strap.",
     8, 175.00),
    (37, "قفازات رفع الأثقال",
     "Leather weight lifting gloves with wrist support and anti-blister palm padding.",
     8, 120.00),
    (38, "حذاء رياضي للجري",
     "Lightweight running shoes with cushioned sole breathable mesh upper and reflective details.",
     8, 480.00),
    (39, "زجاجة ماء رياضية",
     "BPA free insulated sports water bottle 750ml keeps drinks cold 24 hours and hot 12 hours.",
     8, 85.00),
    (40, "طقم مقاومة المطاط",
     "Resistance bands set 5 levels for strength training home workout pilates and physical therapy.",
     8, 145.00),
]

# 15 simulated users
USER_IDS = [
    "user-001", "user-002", "user-003", "user-004", "user-005",
    "user-006", "user-007", "user-008", "user-009", "user-010",
    "user-011", "user-012", "user-013", "user-014", "user-015",
]

SELLER_ID = "seller-default-001"


def _predict_cluster(description: str) -> int:
    """Use the live TF-IDF + KMeans model to predict cluster for a product description."""
    try:
        from ml_api.models.loader import ModelManager
        manager = ModelManager()
        tfidf = manager.get_model("tfidf_vectorizer")
        kmeans = manager.get_model("kmeans_model")
        if tfidf is None or kmeans is None:
            return random.randint(0, 9)
        vec = tfidf.transform([description])
        return int(kmeans.predict(vec)[0])
    except Exception:
        return random.randint(0, 9)


def seed(clear_existing: bool = False) -> dict:
    """
    Insert seed data into the database.

    Args:
        clear_existing: If True, delete existing test data before inserting.

    Returns:
        Summary dict with counts of inserted records.
    """
    init_db()
    db = SessionLocal()

    summary = {
        "categories_inserted": 0,
        "products_inserted": 0,
        "interactions_inserted": 0,
        "embeddings_inserted": 0,
        "skipped": [],
    }

    try:
        # ── Optional cleanup ──────────────────────────────────────────────
        if clear_existing:
            db.query(ProductEmbedding).filter(
                ProductEmbedding.product_id.in_([p[0] for p in PRODUCTS])
            ).delete(synchronize_session=False)
            db.query(UserInteraction).filter(
                UserInteraction.UserId.in_(USER_IDS)
            ).delete(synchronize_session=False)
            db.query(Product).filter(
                Product.Id.in_([p[0] for p in PRODUCTS])
            ).delete(synchronize_session=False)
            db.query(ProductCategory).filter(
                ProductCategory.Id.in_([c["Id"] for c in CATEGORIES])
            ).delete(synchronize_session=False)
            db.commit()

        # ── Categories ────────────────────────────────────────────────────
        for cat in CATEGORIES:
            exists = db.query(ProductCategory).filter(
                ProductCategory.Id == cat["Id"]
            ).first()
            if exists:
                summary["skipped"].append(f"category:{cat['Id']}")
                continue
            db.add(ProductCategory(
                Id=cat["Id"],
                NameEn=cat["NameEn"],
                NameAr=cat["NameAr"],
                Image=cat["Image"],
            ))
            summary["categories_inserted"] += 1
        db.commit()

        # ── Products ──────────────────────────────────────────────────────
        for prod_id, name, description, cat_id, price in PRODUCTS:
            exists = db.query(Product).filter(Product.Id == prod_id).first()
            if exists:
                summary["skipped"].append(f"product:{prod_id}")
                continue
            db.add(Product(
                Id=prod_id,
                NameAr=name,          # seed name is Arabic
                DescriptionEn=description,
                CategoryId=cat_id,
                Price=price,
                ImageUrl=f"https://example.com/images/product-{prod_id}.jpg",
                Quantity=random.randint(10, 200),
                SellerID=SELLER_ID,
            ))
            summary["products_inserted"] += 1
        db.commit()

        # ── User Interactions (ratings) ───────────────────────────────────
        # Each user rates a random subset of products (8–20 products each)
        random.seed(42)
        interaction_id = _next_interaction_id(db)

        for user_id in USER_IDS:
            # Each user rates a random selection of products
            rated_products = random.sample(PRODUCTS, k=random.randint(8, 20))
            for prod_id, *_ in rated_products:
                exists = db.query(UserInteraction).filter(
                    UserInteraction.UserId == user_id,
                    UserInteraction.ProductID == prod_id,
                ).first()
                if exists:
                    continue

                # Bias rating slightly per product to create realistic patterns
                base_rating = random.choices([3, 4, 5], weights=[2, 4, 4])[0]
                rating = max(1, min(5, base_rating + random.randint(-1, 1)))

                db.add(UserInteraction(
                    Id=interaction_id,
                    UserId=user_id,
                    ProductID=prod_id,
                    Rating=rating,
                    IsFavourite=(rating >= 4),
                    InteractionDate=datetime.utcnow() - timedelta(
                        days=random.randint(0, 180)
                    ),
                ))
                interaction_id += 1
                summary["interactions_inserted"] += 1

        db.commit()

        # ── Product Embeddings (cluster assignments) ───────────────────────
        for prod_id, name, description, cat_id, price in PRODUCTS:
            exists = db.query(ProductEmbedding).filter(
                ProductEmbedding.product_id == prod_id
            ).first()
            if exists:
                summary["skipped"].append(f"embedding:{prod_id}")
                continue

            cluster_id = _predict_cluster(description)
            db.add(ProductEmbedding(
                product_id=prod_id,
                cluster_id=cluster_id,
                last_updated=datetime.utcnow(),
            ))
            summary["embeddings_inserted"] += 1

        db.commit()

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Seed failed: {e}") from e
    finally:
        db.close()

    return summary


def _next_interaction_id(db) -> int:
    """Get the next available UserInteraction.Id."""
    from sqlalchemy import func
    max_id = db.query(func.max(UserInteraction.Id)).scalar()
    return (max_id or 0) + 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed test data into the ML API database")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing seed data before inserting")
    args = parser.parse_args()

    print("Seeding database...")
    result = seed(clear_existing=args.clear)

    print(f"  Categories inserted : {result['categories_inserted']}")
    print(f"  Products inserted   : {result['products_inserted']}")
    print(f"  Interactions inserted: {result['interactions_inserted']}")
    print(f"  Embeddings inserted : {result['embeddings_inserted']}")
    if result["skipped"]:
        print(f"  Skipped (already exist): {len(result['skipped'])}")
    print("Done.")
