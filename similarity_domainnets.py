import torch, os, clip, glob, json
from PIL import Image
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import shutil
import numpy as np
import pandas as pd
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


# pip install git+https://github.com/openai/CLIP.git


# load the clip model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)



def Convert(image):
    return image.convert("RGB")

class DatasetWrapper(Dataset):
    def __init__(self, images_path, input_size = 224, eval = "clip"):
        self.images_path = images_path; self.input_size = input_size
        # Build transform
        self.trans = T.Compose([T.Resize(size=(self.input_size, self.input_size)), T.ToTensor()])
        self.trans = T.Compose([
            Resize(input_size, interpolation=Image.BICUBIC),
            CenterCrop(input_size),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

        if eval=="dino":
          self.trans =  Compose([
              Resize(self.input_size, interpolation=Image.BICUBIC),
              CenterCrop(self.input_size),
              Convert,
              ToTensor(),
              Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
          ])
        # total_images = args.images
        # distribution = [i for i in range(total_images)]
        # num_selected_images = int(selection_p * total_images)
        # sampled_elements = random.sample(distribution, num_selected_images)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # print(idx)
        img = preprocess(Image.open(self.images_path[idx]))
        return img


def do_cosine(i_batch, j_batch):
    # print(i_batch.shape, j_batch.shape)

    i_batch = i_batch.unsqueeze(0).expand(j_batch.size(0), -1, -1)  # shape: (8, 4, 12)
    # Compute cosine similarity
    with torch.no_grad():
        cosine_sim = torch.nn.functional.cosine_similarity(i_batch, j_batch.unsqueeze(1), dim=2)  # shape: (8, 4)
    return cosine_sim.mean(dim=1)


def CLIP_I(dir1, dir2, batch_size = 400, matrix="cosine"):

    scores = []
    for i in range(len(dir1)):
        dir1_images = sorted([os.path.join(dir1[i], f) for f in os.listdir(dir1[i])])
        dir2_images = sorted([os.path.join(dir2[i], f) for f in os.listdir(dir2[i])])

        # create dataloader for both the folder
        dir1_datloader = DataLoader(DatasetWrapper(dir1_images), batch_size=batch_size, shuffle=False)
        dir2_datloader = DataLoader(DatasetWrapper(dir2_images), batch_size=batch_size, shuffle=False)

        
        i_batchs, j_batchs = [], []
        for i_batch in dir1_datloader:
            i_batch = model.encode_image(i_batch.to(device)).to(device) # pass this to CLIP model
            # print(i_batch.shape); exit()
            i_batchs.append(i_batch)

        i_batchs = torch.stack(i_batchs)
        i_batch = torch.mean(i_batchs, dim=1); del i_batchs

        for j_batch in dir2_datloader:
            j_batch = model.encode_image(j_batch.to(device)).to(device) # pass this to CLIP model
            j_batchs.append(j_batch)

        j_batchs = torch.stack(j_batchs)
        j_batch = torch.mean(j_batchs, dim=1); del j_batchs

        score = do_cosine(i_batch, j_batch)
        scores.append(score.item())
    
    return np.mean(np.array(scores))
    


def DINO(dir1, dir2, batch_size = 100, matrix="cosine"):

    scores = []
    for i in range(len(dir1)):
        # print("shyam")
        dir1_images = sorted([os.path.join(dir1[i], f) for f in os.listdir(dir1[i])])
        dir2_images = sorted([os.path.join(dir2[i], f) for f in os.listdir(dir2[i])])

        # create dataloader for both the folder
        dir1_datloader = DataLoader(DatasetWrapper(dir1_images), batch_size=batch_size, shuffle=False)
        dir2_datloader = DataLoader(DatasetWrapper(dir2_images), batch_size=batch_size, shuffle=False)

        with torch.no_grad():  # Add this!
            i_batchs, j_batchs = [], []
            for i_batch in dir1_datloader:
                # print(i_batch.shape); exit()
                # print("kamal")
                i_batch = dino_model(i_batch.to(device)).to(device) # pass this to CLIP model
                # print(i_batch.shape); exit()
                i_batchs.append(i_batch)

            i_batchs = torch.vstack(i_batchs)
            i_batch = torch.mean(i_batchs, dim=0, keepdim=True); del i_batchs

            for j_batch in dir2_datloader:
                j_batch = dino_model(j_batch.to(device)).to(device) # pass this to CLIP model
                j_batchs.append(j_batch)

            j_batchs = torch.vstack(j_batchs)
            # print(j_batchs.shape); exit()
            j_batch = torch.mean(j_batchs, dim=0, keepdim=True); del j_batchs

        # print(j_batch.shape, i_batch.shape); exit()
        score = do_cosine(i_batch, j_batch)
        scores.append(score.item())
        del i_batch
        del j_batch
        del dir1_datloader
        del dir2_datloader
        # torch.cuda.empty_cache()

    return np.mean(np.array(scores))


# def DINO(org_folder_path = None, gen_folder_path = None):
#     org_image_files = sorted([os.path.join(org_folder_path, f) for f in os.listdir(org_folder_path)])
#     gen_image_files = sorted([os.path.join(gen_folder_path, f) for f in os.listdir(gen_folder_path)])

#     # create dataloader for both the folder
#     if(len(org_image_files)>128): batch_size = 128
#     else: batch_size = len(org_image_files)
#     org_datloader = DataLoader(DatasetWrapper(org_image_files, eval="dino"), batch_size=batch_size, shuffle=False)
#     if(len(gen_image_files)>128): batch_size = 128
#     else: batch_size = len(gen_image_files)
#     gen_dataloader = DataLoader(DatasetWrapper(gen_image_files, eval="dino"), batch_size=batch_size, shuffle=False)

#     clipi = []
#     dino_model.eval()
#     for i_batch in org_datloader:
#         i_batch = dino_model(i_batch.to(device)).to(device) # pass this to CLIP model
#     for j_batch in gen_dataloader:
#         j_batch = dino_model(j_batch.to(device)).to(device) # pass this to CLIP model

#     i_batch = i_batch.unsqueeze(0).expand(j_batch.size(0), -1, -1)  # shape: (8, 4, 12)

#     # Compute cosine similarity
#     with torch.no_grad():
#         cosine_sim = torch.nn.functional.cosine_similarity(i_batch, j_batch.unsqueeze(1), dim=2)  # shape: (8, 4)
#     return cosine_sim.mean(dim=1)


def evaluator(dir1, dir2, model_backbone="clip"):
    if model_backbone=="clip":
        score = CLIP_I(dir1=dir1, dir2=dir2)
    elif model_backbone=="dino":
        score = DINO(dir1=dir1, dir2=dir2)
    else:
        raise ValueError(f"wrong MODEL backbone")

    return score
    
    # save the score as a CSV file with class_names
    # class_names[0] + '_' + class_names[1]




painting_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/painting"
clipart_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/clipart"
real_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/real"
infograph_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/infograph"
quickdraw_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/quickdraw"
sketch_path = "/home/test/Documents/sampath/flda/data/DomainNetS/raw/sketch"



all_classes = os.listdir(painting_path)
painting_class = [os.path.join(painting_path, i) for i in all_classes]


assert len(os.listdir(painting_path))==len(os.listdir(clipart_path))==len(os.listdir(real_path))==len(os.listdir(infograph_path))==len(os.listdir(quickdraw_path))==len(os.listdir(sketch_path))
clipart_class = [os.path.join(clipart_path, i) for i in all_classes]
real_class = [os.path.join(real_path, i) for i in all_classes]
infograph_class = [os.path.join(infograph_path, i) for i in all_classes]
quickdraw_class = [os.path.join(quickdraw_path, i) for i in all_classes]
sketch_class = [os.path.join(sketch_path, i) for i in all_classes]

def class_wise(class_names=['painting', 'clipart'], model_backbone="clip"):
    assert len(class_names)==2
    paths = []
    for i in range(len(class_names)):
        if class_names[i]=='painting':
            paths.append(painting_class)
        elif class_names[i]=='clipart':
            paths.append(clipart_class)
        elif class_names[i]=='real':
            paths.append(real_class)
        elif class_names[i]=='infograph':
            paths.append(infograph_class)
        elif class_names[i]=='quickdraw':
            paths.append(quickdraw_class)
        elif class_names[i]=='sketch':
            paths.append(sketch_class)

    
    assert len(paths[0])==len(paths[1])
    score = evaluator(paths[0], paths[1], model_backbone=model_backbone)
    return score

# score_art_clipart = class_wise(class_names=['art', 'clipart'])
# score_art_product = class_wise(class_names=['art', 'product'])
# score_art_real = class_wise(class_names=['art', 'real'])
# score_clipart_real = class_wise(class_names=['clipart', 'real'])
# score_product_real = class_wise(class_names=['product', 'real'])
# score_product_clipart = class_wise(class_names=['product', 'clipart'])

# print(score_art_clipart)        # 0.9002629206730769
# print(score_art_product)        # 0.8438176081730769
# print(score_art_real)           # 0.9316781850961539
# print(score_clipart_real)       # 0.8946814903846154
# print(score_product_real)       # 0.9251427283653846
# print(score_product_clipart)    # 0.8445537860576923

# """
#             Art     clipart     product     real
# Art         1       0.90        0.844       0.932
# clipart     0.90    1           0.8445      0.895
# product     0.844   0.8445      1           0.925
# real        0.932   0.895       0.925       1
# """


score_painting_clipart = class_wise(class_names=['painting', 'clipart'], model_backbone="dino")
score_painting_infograph = class_wise(class_names=['painting', 'infograph'], model_backbone="dino")
score_painting_real = class_wise(class_names=['painting', 'real'], model_backbone="dino")
score_painting_quickdraw = class_wise(class_names=['painting', 'quickdraw'], model_backbone="dino")
score_painting_sketch = class_wise(class_names=['painting', 'sketch'], model_backbone="dino")

score_sketch_quickdraw = class_wise(class_names=['quickdraw', 'sketch'], model_backbone="dino")
score_sketch_clipart = class_wise(class_names=['clipart', 'sketch'], model_backbone="dino")
score_sketch_real = class_wise(class_names=['real', 'sketch'], model_backbone="dino")
score_sketch_infograph = class_wise(class_names=['infograph', 'sketch'], model_backbone="dino")

score_quickdraw_real = class_wise(class_names=['quickdraw', 'real'], model_backbone="dino")
score_quickdraw_infograph = class_wise(class_names=['quickdraw', 'infograph'], model_backbone="dino")
score_quickdraw_clipart = class_wise(class_names=['quickdraw', 'clipart'], model_backbone="dino")

score_real_infograph = class_wise(class_names=['real', 'infograph'], model_backbone="dino")
score_real_clipart = class_wise(class_names=['real', 'clipart'], model_backbone="dino")

score_clipart_infograph = class_wise(class_names=['clipart', 'infograph'], model_backbone="dino")

print(score_painting_clipart)
print(score_painting_infograph)
print(score_painting_real)
print(score_painting_quickdraw)
print(score_painting_sketch)

print(score_sketch_quickdraw)
print(score_sketch_clipart)
print(score_sketch_real)
print(score_sketch_infograph)

print(score_quickdraw_real)
print(score_quickdraw_infograph)
print(score_quickdraw_clipart)

print(score_real_infograph)
print(score_real_clipart)

print(score_clipart_infograph)
"""
            Art     clipart     product     real
Art         1       0.61        0.71        0.84
clipart     0.61    1           0.69        0.62
product     0.71    0.69        1           0.83
real        0.84    0.62        0.83        1
"""
