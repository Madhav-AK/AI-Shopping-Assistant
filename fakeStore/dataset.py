import os
import shutil
import json
import random

# Paths
source_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ut-zap50k-images-square"))
destination_folder = os.path.join(os.path.dirname(__file__), "static", "uploads", "dataset")
os.makedirs(destination_folder, exist_ok=True)

# Data holders
metadata = []
filename_to_id = {}
current_id = 1

# Traverse source folder
for root, dirs, files in os.walk(source_folder):
    for file in sorted(files):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Extract category / subcategory / brand
                path_parts = root.replace(source_folder, "").strip(os.sep).split(os.sep)
                if len(path_parts) != 3:
                    print(f"⚠ Skipping: Unexpected folder depth: {root}")
                    continue

                category, subcategory, brand = path_parts

                # Paths
                new_filename = f"{current_id}.jpg"
                src_path = os.path.join(root, file)
                dst_path = os.path.join(destination_folder, new_filename)

                # Copy file
                shutil.copyfile(src_path, dst_path)

                # Add to metadata
                metadata.append({
                    "id": current_id,
                    "brand": brand,
                    "category": category,
                    "subcategory": subcategory,
                    "price_cents": random.randint(1000, 10000),
                    "description": f"{brand} stylish {subcategory.lower()} from our {category.lower()} range.",
                    "image_path": f"/static/uploads/dataset/{new_filename}",
                    "filename": file
                })

                # Add to filename-to-id map (optional: relative path as key)
                rel_path = os.path.relpath(os.path.join(root, file), source_folder)
                filename_to_id[rel_path] = current_id

                current_id += 1
                if current_id % 1000 == 0:
                    print(f"Processed {current_id} files...")

            except Exception as e:
                print(f"❌ Failed on {file}: {e}")

# Define path to metadata.json inside static/uploads
metadata_output_path = os.path.join(os.path.dirname(__file__), "static", "uploads", "metadata.json")
# Save metadata
with open(metadata_output_path, "w") as f:
    json.dump(metadata, f, indent=4)


#with open("/home/specapoorv/test2/AI-Shopping-Assistant/fakeStore/filename-id.json", "w") as f:
 #   json.dump(filename_to_id, f, indent=2)

print(f"✅ Done. Processed {len(metadata)} products.")