import os
import numpy as np
from tqdm import tqdm

npz_path = "/data4/CheXpert/"

dict1 = np.load(npz_path + "train.npz")
dict2 = np.load(npz_path + "validation.npz")

out_dir = '/data4/CheXpert/chexpert_embeddings'

# Combine both dictionaries into a list of dictionaries
datasets = [("train", dict1), ("validation", dict2)]

for dataset_name, data_dict in datasets:
    tqdm_iterator = tqdm(data_dict.items(), total=len(data_dict), desc=f"Saving {dataset_name} embeddings")
    
    for key, value in tqdm_iterator:
        split = key.split("/")
        preproc_filename = f"{split[2]}_{split[3]}_{split[4].replace('.jpg', '.dat')}"
        out_path = os.path.join(out_dir, preproc_filename)
        with open(out_path, "wb") as file:
            file.write(value.tobytes())
