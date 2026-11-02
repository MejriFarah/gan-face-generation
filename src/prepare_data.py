import os
import shutil
import random

source_folder=r"C:\Users\emili\Downloads\archive (3)\img_align_celeba\img_align_celeba"
destination_folder="data/celebs-subset"
num_img=50000
images=[
    file for file in os.listdir(source_folder)
    if file.endswith(".jpg")
]
print(f"There exists {len(images)} celebA images.")
random.seed(42)
selected_images=random.sample(images,num_img)
for image in selected_images:
    source=os.path.join(source_folder,image)
    destination=os.path.join(destination_folder,image)
    shutil.copy(source,destination)
   