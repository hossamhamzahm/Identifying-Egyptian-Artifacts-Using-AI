import os
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image



root_path = "/mnt/c/Users/hosam/Desktop/grad/datasets"
org_dir_name = "2nd_org"


original_image_path = os.path.join(root_path, org_dir_name)
augmented_image_path = os.path.join(root_path, org_dir_name + "_augmented")



# Example transformations
transform = transforms.Compose([
    # transforms.RandomCrop((224, 224)),  # Resize the image to (224, 224)
    transforms.ColorJitter(brightness=0.6, hue=(-0.5, 0.5), contrast=0.1),
    transforms.RandomRotation(degrees=(-40, 40)),
    transforms.Resize((576, 567)),  # Resize the image to (224, 224)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomChoice([
        transforms.Pad(padding=10),
        transforms.CenterCrop((432, 400)),
        transforms.RandomRotation(30),
    ]),
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.6, 0.6, 0.6]),  # Normalize the image
])




def create_dir(read_path, write_path):    
    # make root directory
    os.mkdir(write_path)
    
    for dir_name in os.listdir(read_path):
        try:
            # make sub-directory for each class
            os.mkdir(os.path.join(write_path, dir_name))
        except Exception:
            pass




def create_augment_imgs(read_path, write_path):
    # dir_num = 0
    for dir_name in os.listdir(read_path):
        # dir_num += 1
        # if dir_num < 11: continue
        # try:
        img_num = 0

        final_read_path = os.path.join(read_path, dir_name)
        final_write_path = os.path.join(write_path, dir_name)

        for image_name in os.listdir(final_read_path):
            image = Image.open(os.path.join(final_read_path, image_name)).convert('RGB')
            # print(os.path.join(dir_path, image_path))
            # Apply transformations

            for _ in range(300):
                m = random.uniform(0.3, 0.6)
                img_num += 1
                
                save_image(transform(image), os.path.join(final_write_path, str(img_num) + ".png"))



def load_imgs(dir_path):
    imgs_tensors = []

    for image_path in os.listdir(dir_path):
        try:
            image = Image.open(os.path.join(dir_path, image_path))
            # print(os.path.join(dir_path, image_path))
            # Apply transformations
            image_tensor = transform(image)
            imgs_tensors.append((image_tensor))
        except Exception:
            pass
    
    return imgs_tensors



create_dir(original_image_path, augmented_image_path)
create_augment_imgs(original_image_path, augmented_image_path)
