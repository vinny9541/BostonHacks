import os
import random

# Path to the directory containing subfolders with images

main_directory = 'garbage_classification'

# Get a list of all jpg images
all_images = []
for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.lower().endswith('.jpg'):
            all_images.append(os.path.join(root, file))

# Randomly shuffle the list
random.shuffle(all_images)

# Calculate how many images to remove
images_to_remove = len(all_images) - 2500
print(images_to_remove)

if images_to_remove > 0:
    # Remove the images
    for i in range(images_to_remove):
        os.remove(all_images[i])
        print(f"Deleted {all_images[i]}")
else:
    print("There are already less than or equal to 2500 images.")


# Make sure to check the script on a small test folder first to ensure it works as expected!
