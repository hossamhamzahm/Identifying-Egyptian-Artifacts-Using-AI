import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd


# Example transformations
transform = transforms.Compose([
    transforms.Resize((450, 450)),  # Resize the image to (224, 224)
    transforms.ToTensor(),          # Convert the PIL image to a PyTorch tensor (0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])


def load_imgs(dir_path):
    imgs_tensors = []

    for image_path in os.listdir(dir_path):
        try:
            image = Image.open(os.path.join(dir_path, image_path))
            # Apply transformations
            image_tensor = transform(image)
            imgs_tensors.append((image_tensor))
            
            return imgs_tensors
        except Exception:
            pass

traing_image_path = "/mnt/c/Users/hosam/Desktop/grad/datasets/monuments/train"

imgs_tensors = load_imgs(traing_image_path)
batch_tensor = torch.stack(imgs_tensors)


df = pd.read_csv(os.path.join(traing_image_path, "_classes.csv"))
classification = {}

dirs = {}

for idx in df.index:
    # df.loc[idx][1:] = pd.to_numeric(df.loc[idx][1:]))
    try:
        # print((pd.to_numeric(df.loc[idx][1:]).idxmax()).strip())
        classification[df.loc[idx][0]] = pd.to_numeric(df.loc[idx][1:]).idxmax()
        os.rename(
            os.path.join(traing_image_path, df.loc[idx][0]), 
            os.path.join(traing_image_path, pd.to_numeric(df.loc[idx][1:]).idxmax().strip(), df.loc[idx][0])
        )
    except Exception:
        pass
    # dirs[pd.to_numeric(df.loc[idx][1:]).idxmax()] = 1


# create dirs
# for dir in dirs:
    # os.mkdir(os.path.join(traing_image_path, dir.strip()))
