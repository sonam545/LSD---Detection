import os
import cv2

source_base = '.'  

dest_base = 'clean_data'
healthy_dir = os.path.join(dest_base, 'healthy')
infected_dir = os.path.join(dest_base, 'infected')

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(infected_dir, exist_ok=True)

IMG_SIZE = (224, 224)
healthy_count = 0
infected_count = 0

healthy_folders = [
    'archive/Lumpy Skin Images Dataset/Normal Skin',
    'archive2/healthycows',
    'Normal Skin'
]

infected_folders = [
    'archive/Lumpy Skin Images Dataset/Lumpy Skin',
    'archive2/lumpycows',
    'Lumpy Skin'
]

# Function to process images
def process_images(folder_list, dest_folder, prefix, count):
    for folder in folder_list:
        folder_path = os.path.join(source_base, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(folder_path, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.resize(img, IMG_SIZE)
                        save_path = os.path.join(dest_folder, f"{prefix}_{count}.jpg")
                        cv2.imwrite(save_path, img)
                        count += 1
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
    return count

# Process images from both classes
healthy_count = process_images(healthy_folders, healthy_dir, "healthy", healthy_count)
infected_count = process_images(infected_folders, infected_dir, "infected", infected_count)

print(f"✅ Finished! {healthy_count} healthy images and {infected_count} infected images.")
