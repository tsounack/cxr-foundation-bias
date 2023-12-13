from cxr_foundation.inference import generate_embeddings, InputFileType, OutputFileType, ModelVersion
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from PIL import Image

EMBEDDINGS_DIR = "/data4/CheXpert/chexpert_embedding_normal"
MODEL_VERSION = ModelVersion.V1
data_dir = '../datafiles/chexpert/'
img_data_dir = '/data4/CheXpert/'

df_test = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test['Path'] = df_test['Path'].apply(lambda x: img_data_dir + x)

first_image = Image.open(df_test.iloc[2, 0])
print(first_image.size)

df_img_data = pd.read_csv(data_dir + 'chexpert.sample.csv')
df_img_data['path_preproc'] = df_img_data['path_preproc'].apply(lambda x: img_data_dir + x)

import logging
logging.getLogger('google.auth._default').setLevel(logging.ERROR)

generate_embeddings(input_files=df_img_data["path_preproc"].values, output_dir=EMBEDDINGS_DIR,
    input_type=InputFileType.PNG, output_type=OutputFileType.NPZ, model_version=MODEL_VERSION)