import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

data_dir = '../datafiles/chexpert/'
df_demo = pd.DataFrame(pd.read_excel(data_dir + 'CHEXPERT DEMO.xlsx', engine='openpyxl'))

df_demo = df_demo.rename(columns={'PRIMARY_RACE': 'race'})
df_demo = df_demo.rename(columns={'PATIENT': 'patient_id'})
df_demo = df_demo.rename(columns={'GENDER': 'sex'})
df_demo = df_demo.rename(columns={'AGE_AT_CXR': 'age'})
df_demo = df_demo.rename(columns={'ETHNICITY': 'ethnicity'})
df_demo = df_demo.drop(['sex', 'age'], axis=1)
df_demo.head()

df_data_split = pd.read_csv(data_dir + 'chexpert_split_2021_08_20.csv').set_index('index')

df_img_data = pd.read_csv(data_dir + 'train.csv')

df_img_data = pd.concat([df_img_data,df_data_split], axis=1)
df_img_data = df_img_data[~df_img_data.split.isna()]

split =  df_img_data.Path.str.split("/", expand = True)
df_img_data["patient_id"] = split[2]
df_img_data = df_img_data.rename(columns={'Age': 'age'})
df_img_data = df_img_data.rename(columns={'Sex': 'sex'})
df_img_data.head()

df_cxr = df_demo.merge(df_img_data, on="patient_id")
df_cxr.head()

white = 'White'
asian = 'Asian'
black = 'Black'

mask = (df_cxr.race.str.contains("Black", na=False))
df_cxr.loc[mask, "race"] = black

mask = (df_cxr.race.str.contains("White", na=False))
df_cxr.loc[mask, "race"] = white

mask = (df_cxr.race.str.contains("Asian", na=False))
df_cxr.loc[mask, "race"] = asian

df_cxr['race'].unique()

df_cxr = df_cxr[df_cxr.race.isin([asian,black,white])]

df_cxr = df_cxr[df_cxr.ethnicity.isin(["Non-Hispanic/Non-Latino","Not Hispanic"])]

df_cxr = df_cxr[df_cxr["Frontal/Lateral"]=="Frontal"]

df_cxr['race_label'] = df_cxr['race']

df_cxr.loc[df_cxr['race_label'] == white, 'race_label'] = 0
df_cxr.loc[df_cxr['race_label'] == asian, 'race_label'] = 1
df_cxr.loc[df_cxr['race_label'] == black, 'race_label'] = 2

df_cxr['sex_label'] = df_cxr['sex']

df_cxr.loc[df_cxr['sex_label'] == 'Male', 'sex_label'] = 0
df_cxr.loc[df_cxr['sex_label'] == 'Female', 'sex_label'] = 1

labels = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

df_cxr['disease'] = df_cxr[labels[0]]
df_cxr.loc[df_cxr[labels[0]] == 1, 'disease'] = labels[0]
df_cxr.loc[df_cxr[labels[10]] == 1, 'disease'] = labels[10]
df_cxr.loc[df_cxr['disease'].isna(), 'disease'] = 'Other'

df_cxr['disease_label'] = df_cxr['disease']
df_cxr.loc[df_cxr['disease_label'] == labels[0], 'disease_label'] = 0
df_cxr.loc[df_cxr['disease_label'] == labels[10], 'disease_label'] = 1
df_cxr.loc[df_cxr['disease_label'] == 'Other', 'disease_label'] = 2

# point to the parent directory that contains the folder 'CheXpert-v1.0'
img_data_dir = '/data4/CheXpert/'

from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from multiprocessing import Pool
from functools import partial

df_cxr['path_preproc'] = df_cxr['Path']

preproc_dir = 'preproc_224x224/'
out_dir = img_data_dir

if not os.path.exists(out_dir + preproc_dir):
    os.makedirs(out_dir + preproc_dir)


def process_image(idx, df_cxr, preproc_dir, out_dir, img_data_dir):
    p = df_cxr['Path'][idx]
    split =  p.split("/")
    preproc_filename = split[2] + '_' + split[3] + '_' + split[4]
    df_cxr.loc[idx, 'path_preproc'] = preproc_dir + preproc_filename
    out_path = out_dir + preproc_dir + preproc_filename
    
    if not os.path.exists(out_path):
        image = imread(img_data_dir + p)
        image = resize(image, output_shape=(224, 224), preserve_range=True)
        imsave(out_path, image.astype(np.uint8))

df_cxr_copy = df_cxr.copy()

with Pool() as pool:
    func = partial(process_image, df_cxr=df_cxr_copy, preproc_dir=preproc_dir, out_dir=out_dir, img_data_dir=img_data_dir)
    for _ in tqdm(pool.imap_unordered(func, range(len(df_cxr_copy['Path'])-1, -1, -1)), total=len(df_cxr_copy)):
        pass

df_cxr_copy.to_csv(data_dir + 'chexpert.sample.csv')
