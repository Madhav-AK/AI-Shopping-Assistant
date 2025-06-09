from app import app, db
from app.db_models import Item


with app.app_context():
    # Clear existing items from the database
    num_deleted = Item.query.delete()
    db.session.commit()
    print(f"✅ Cleared {num_deleted} items from the database.")