import json
from app import app, db  # Make sure app is also imported
from app.db_models import Item
import os

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the metadata.json
json_path = os.path.join(BASE_DIR, '..', 'static', 'uploads', 'metadata.json')
json_path = os.path.normpath(json_path)  # Normalize the path (e.g., convert ..)

with app.app_context():  # <- FIX: Start application context
    with open(json_path) as f:
        data = json.load(f)

    for entry in data:
        item = Item(
            id=entry['id'],
            brand=entry['brand'],
            category=entry['category'],
            subcategory=entry.get('subcategory'),
            price_cents=entry['price_cents'],
            description=entry.get('description', ''),
            image_path=entry['image_path']
        )
        db.session.add(item)

    db.session.commit()
    print("Items added successfully.")