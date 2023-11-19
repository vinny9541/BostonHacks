import os
import shutil

# Define the path to the main directory
main_directory = 'trash_or_recycle_classification'

# Subfolders for 'trash' and 'recycle' classification
trash_folders = ['battery', 'biological', 'clothes', 'shoes', 'trash']
recycle_folders = ['brown-glass', 'cardboard', 'green-glass', 'metal', 'paper', 'plastic', 'white-glass']

# Define the target folders for 'trash' and 'recycle'
trash_target_folder = os.path.join(main_directory, 'trash')
recycle_target_folder = os.path.join(main_directory, 'recycle')

# Create target folders if they don't exist
os.makedirs(trash_target_folder, exist_ok=True)
os.makedirs(recycle_target_folder, exist_ok=True)

# Function to move files from source to target folder with conflict resolution
def move_files_to_folder(source_folder, target_folder):
    for file_name in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, file_name)
        target_file_path = os.path.join(target_folder, file_name)

        # Check if the file already exists in the target folder
        if os.path.isfile(target_file_path):
            # Generate a new file name to avoid overwriting
            base, extension = os.path.splitext(file_name)
            counter = 1
            new_file_name = f"{base}_{counter}{extension}"
            new_target_file_path = os.path.join(target_folder, new_file_name)
            
            # Increment the counter until a new unique name is found
            while os.path.isfile(new_target_file_path):
                counter += 1
                new_file_name = f"{base}_{counter}{extension}"
                new_target_file_path = os.path.join(target_folder, new_file_name)
            
            # Move and rename the file to avoid collision
            shutil.move(source_file_path, new_target_file_path)
        else:
            # Move the file if there is no collision
            shutil.move(source_file_path, target_file_path)

# Move files to 'trash' folder
for folder in trash_folders:
    source_folder = os.path.join(main_directory, folder)
    move_files_to_folder(source_folder, trash_target_folder)

# Move files to 'recycle' folder
for folder in recycle_folders:
    source_folder = os.path.join(main_directory, folder)
    move_files_to_folder(source_folder, recycle_target_folder)

print("Classification of images into 'trash' and 'recycle' folders is complete.")
