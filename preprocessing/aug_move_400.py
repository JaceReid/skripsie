import os
import random
import shutil

# ==== CONFIG ====
SOURCE_DIR = "./Datasets/Raw-audio/aug-test/FD_4.0_clipped"
DEST_DIR = "./Datasets/Raw-audio/aug-test/Final/"
NUM_FILES_PER_NAME = 400
# ===============

os.makedirs(DEST_DIR, exist_ok=True)

all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]

# Group files by name before the first underscore
grouped = {}
for filename in all_files:
    key = filename.split('_')[0]
    grouped.setdefault(key, []).append(filename)

# Process each group
for key, files in grouped.items():
    random.shuffle(files)  # Random order
    files_to_move = files[:NUM_FILES_PER_NAME]  # Take first 400

    for file in files_to_move:
        src_path = os.path.join(SOURCE_DIR, file)
        # dst_folder = os.path.join(DEST_DIR)
        dst_folder=DEST_DIR
        os.makedirs(dst_folder, exist_ok=True)  # Keep subfolders by name
        dst_path = os.path.join(dst_folder, file)
        shutil.copy(src_path, dst_path)

    print(f"Moved {len(files_to_move)} files for '{key}'")

print("Done!")
