import os
import shutil
import numpy as np

np.random.seed(42)  # for reproducibility

# Assume data_dir is your original directory with two subdirectories 'recycle' and 'trash'
data_dir = 'trash_or_recycle_classification'
classes = ['recycle', 'trash']

# Define your split ratio
split_train = 0.70
split_val = 0.15
split_test = 0.15

for cls in classes:
    # Create directories
    os.makedirs(f'train/{cls}', exist_ok=True)
    os.makedirs(f'validation/{cls}', exist_ok=True)
    os.makedirs(f'test/{cls}', exist_ok=True)
    
    # Get a list of pictures
    all_files = os.listdir(os.path.join(data_dir, cls))
    np.random.shuffle(all_files)
    
    # Split files
    train_files, val_files, test_files = np.split(np.array(all_files),
                                                  [int(len(all_files)*split_train), 
                                                   int(len(all_files)*(split_train+split_val))])
    
    # Copy files
    for file in train_files:
        shutil.copy(os.path.join(data_dir, cls, file), os.path.join('train', cls, file))
    for file in val_files:
        shutil.copy(os.path.join(data_dir, cls, file), os.path.join('validation', cls, file))
    for file in test_files:
        shutil.copy(os.path.join(data_dir, cls, file), os.path.join('test', cls, file))
