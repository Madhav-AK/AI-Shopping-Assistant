import json
from app import app, db  # Make sure app is also imported
from app.db_models import Item

with app.app_context():  # <- FIX: Start application context
    with open(r'C:\Users\madha\OneDrive\Madhav\IITM\AI Guild\Appian\Appian Round 3\AI-Shopping-Assistant\fakeStore\static\uploads\metadata.json') as f:
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