"""
Script to initialize default admin user
Run this script to create the default admin user if it doesn't exist.

Usage:
    python -m ml_api.init_admin
"""
from sqlalchemy.orm import Session
from .database import SessionLocal, init_db
from .auth.models import User
from .auth.security import get_password_hash
from . import config


def create_admin_user(db: Session) -> None:
    """
    Create default admin user if it doesn't exist.

    Args:
        db: Database session
    """
    # Check if admin user already exists
    existing_user = db.query(User).filter(User.username == config.ADMIN_USERNAME).first()

    if existing_user:
        print(f"Admin user '{config.ADMIN_USERNAME}' already exists")
        return

    # Create admin user
    admin_user = User(
        username=config.ADMIN_USERNAME,
        email=config.ADMIN_EMAIL,
        hashed_password=get_password_hash(config.ADMIN_PASSWORD),
        is_admin=True,
        is_active=True
    )

    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)

    print(f"✓ Created admin user: {admin_user.username} (ID: {admin_user.id})")
    print(f"  Email: {admin_user.email}")
    print(f"  Password: {config.ADMIN_PASSWORD}")
    print("\n⚠️  IMPORTANT: Change the admin password after first login!")


def main():
    """Main function to initialize database and create admin user"""
    print("Initializing database...")
    init_db()
    print("✓ Database initialized")

    print("\nCreating admin user...")
    db = SessionLocal()
    try:
        create_admin_user(db)
    finally:
        db.close()

    print("\n✓ Admin initialization complete")


if __name__ == "__main__":
    main()
