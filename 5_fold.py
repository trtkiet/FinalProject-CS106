import os
import shutil
from sklearn.model_selection import KFold

# Step 1: List all video files
video_folder = 'input_folder/vsumme'
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

video_files = [f for f in os.listdir(video_folder)
               if os.path.splitext(f)[1].lower() in video_extensions]

video_paths = [os.path.join(video_folder, f) for f in video_files]

# Step 2: Setup K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: For each fold, copy files into separate folders
output_base = 'input_folder/vsumme'  # Where all folds will be stored
os.makedirs(output_base, exist_ok=True)

for fold_idx, (train_index, val_index) in enumerate(kf.split(video_paths)):
    fold_name = f'fold_{fold_idx + 1}'
    train_folder = os.path.join(output_base, fold_name, 'train')
    val_folder = os.path.join(output_base, fold_name, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Copy training videos
    for i in train_index:
        src = video_paths[i]
        dst = os.path.join(train_folder, os.path.basename(src))
        shutil.copy(src, dst)

    # Copy validation videos
    for i in val_index:
        src = video_paths[i]
        dst = os.path.join(val_folder, os.path.basename(src))
        shutil.copy(src, dst)

    print(f"Fold {fold_idx + 1}: {len(train_index)} train, {len(val_index)} val")
