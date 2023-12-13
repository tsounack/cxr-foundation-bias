import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from skimage.io import imread
from tqdm import tqdm


def createCheXpertDataset(hparams, img_data_dir, data_dir, dict1_path, dict2_path):
    class CheXpertDataset(Dataset):
        def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True):
            self.data = pd.read_csv(csv_file_img)
            # self.data = self.data.sample(frac=0.5, random_state=42)
            # self.data = self.data.reset_index(drop=True)
            self.image_size = image_size
            self.do_augment = augmentation
            self.pseudo_rgb = pseudo_rgb
            self.data_dir = data_dir

            self.labels = [
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

            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
            ])

            self.dict1_path = dict1_path
            self.dict2_path = dict2_path

            self.samples = self.generate_samples(img_data_dir=img_data_dir)

        def __len__(self):
            return len(self.data)
        
        #### TASK SPECIFIC FUNCTIONS ####
        
        if hparams.mode == 'baseline':
            if hparams.extractor == 'densenet':
                def generate_samples(self, **kwargs):
                    samples = []
                    for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
                        img_path = kwargs['img_data_dir'] + self.data.loc[idx, 'path_preproc']
                        img_label = np.zeros(len(self.labels), dtype='float32')
                        for i in range(0, len(self.labels)):
                            img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')
                        sample = {'image_path': img_path, 'label': img_label}
                        samples.append(sample)
                    return samples
                
                def __getitem__(self, item):
                    sample = self.get_sample(item)
                    image = torch.from_numpy(sample['image']).unsqueeze(0)
                    label = torch.from_numpy(sample['label'])
                    if self.do_augment:
                        image = self.augment(image)
                    if self.pseudo_rgb:
                        try:
                            image = image.repeat(3, 1, 1)
                        except Exception as e:
                            image = image.permute(0, 3, 1, 2).squeeze(0)
                    return {'image': image, 'label': label}
                    
                def get_sample(self, item):
                    sample = self.samples[item]
                    image = imread(sample['image_path']).astype(np.float32)
                    return {'image': image, 'label': sample['label']}

            elif hparams.extractor == 'foundation':
                def generate_samples(self, **kwargs):
                    samples = []
                    dict1 = np.load(self.dict1_path, allow_pickle=True)
                    dict2 = np.load(self.dict2_path, allow_pickle=True)
                    for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
                        img_path = self.data.loc[idx, 'path_preproc']
                        img_label = np.array([self.data.loc[idx, label.strip()] == 1 for label in self.labels], dtype='float32')
                        img_fn = os.path.basename(img_path)
                        parts = img_fn.split('_')
                        patient_str, study_str, view_str, view_type = parts[0][7:], parts[1][5:], parts[2][4:], parts[3][:-4]
                        formatted_filename = f"patient{int(patient_str):05d}/study{int(study_str)}/view{view_str}_{view_type}.jpg"
                        key1 = "CheXpert-v1.0/train/" + formatted_filename
                        key2 = "CheXpert-v1.0/valid/" + formatted_filename
                        features = dict1.get(key1, dict2.get(key2)).astype(np.float32)
                        sample = {'image_path': img_path, 'label': img_label, 'features': features}
                        samples.append(sample)
                    return samples

                def __getitem__(self, item):
                    sample = self.samples[item]
                    features = torch.from_numpy(sample['features'])
                    label = torch.from_numpy(sample['label'])
                    return {'features': features, 'label': label}
                
            else:
                raise NotImplementedError
        
        elif hparams.mode == 'last':
            def get_race_index(self, race):
                race_index = 0
                if race == "White":
                    race_index = 0
                elif race == "Black":
                    race_index = 1
                elif race == "Asian":
                    race_index = 2
                return race_index

            if hparams.extractor == 'densenet':
                def generate_samples(self, **kwargs):
                    samples = []
                    for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
                        img_path = kwargs['img_data_dir'] + self.data.loc[idx, 'path_preproc']
                        img_race = self.data.loc[idx, 'race']
                        img_label = np.zeros(len(self.labels), dtype='float32')
                        for i in range(0, len(self.labels)):
                            img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')
                        sample = {'image_path': img_path, 'label': img_label, 'race': img_race}
                        samples.append(sample)
                    return samples
                
                def __getitem__(self, item):
                    sample = self.get_sample(item)
                    image = torch.from_numpy(sample['image']).unsqueeze(0)
                    label = torch.from_numpy(sample['label'])
                    race_index = self.get_race_index(sample['race'])
                    race = torch.tensor([race_index])
                    if self.do_augment:
                        image = self.augment(image)
                    if self.pseudo_rgb:
                        try:
                            image = image.repeat(3, 1, 1)
                        except Exception as e:
                            image = image.permute(0, 3, 1, 2).squeeze(0)
                    return {'image': image, 'label': label, 'race': race}
                    
                def get_sample(self, item):
                    sample = self.samples[item]
                    image = imread(sample['image_path']).astype(np.float32)
                    return {'image': image, 'label': sample['label'], 'race': sample['race']}
            
            elif hparams.extractor == 'foundation':
                def generate_samples(self, **kwargs):
                    samples = []
                    dict1 = np.load(self.dict1_path, allow_pickle=True)
                    dict2 = np.load(self.dict2_path, allow_pickle=True)
                    for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
                        img_path = self.data.loc[idx, 'path_preproc']
                        img_race = self.data.loc[idx, 'race']
                        img_label = np.array([self.data.loc[idx, label.strip()] == 1 for label in self.labels], dtype='float32')
                        img_fn = os.path.basename(img_path)
                        parts = img_fn.split('_')
                        patient_str, study_str, view_str, view_type = parts[0][7:], parts[1][5:], parts[2][4:], parts[3][:-4]
                        formatted_filename = f"patient{int(patient_str):05d}/study{int(study_str)}/view{view_str}_{view_type}.jpg"
                        key1 = "CheXpert-v1.0/train/" + formatted_filename
                        key2 = "CheXpert-v1.0/valid/" + formatted_filename
                        features = dict1.get(key1, dict2.get(key2)).astype(np.float32)
                        sample = {'image_path': img_path, 'label': img_label, 'features': features, 'race': img_race}
                        samples.append(sample)
                    return samples

                def __getitem__(self, item):
                    sample = self.samples[item]
                    features = torch.from_numpy(sample['features'])
                    label = torch.from_numpy(sample['label'])
                    race_index = self.get_race_index(sample['race'])
                    race = torch.tensor([race_index])
                    return {'features': features, 'label': label, 'race': race}

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
        
    return CheXpertDataset


def createCheXpertDataModule(hparams, **kwargs):
    if hparams.sampler != 'weighted':
        class CheXpertDataModule(pl.LightningDataModule):
            def __init__(self, **kwargs):
                super().__init__()
                self.image_size = kwargs['image_size']
                self.batch_size = kwargs['batch_size']
                self.num_workers = kwargs['num_workers']
                self.train_set = kwargs['train_set']
                self.val_set = kwargs['val_set']
                self.test_set = kwargs['test_set']
                print('#train: ', len(self.train_set))
                print('#val:   ', len(self.val_set))
                print('#test:  ', len(self.test_set))
                self.num_races = kwargs['num_races']
            def train_dataloader(self):
                return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
            def val_dataloader(self):
                return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
            def test_dataloader(self):
                return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    elif hparams.sampler == 'weighted':
        class CheXpertDataModule(pl.LightningDataModule):
            def __init__(self, **kwargs):
                super().__init__()
                self.image_size = kwargs['image_size']
                self.batch_size = kwargs['batch_size']
                self.num_workers = kwargs['num_workers']
                self.train_set = kwargs['train_set']
                self.val_set = kwargs['val_set']
                self.test_set = kwargs['test_set']
                print('#train: ', len(self.train_set))
                print('#val:   ', len(self.val_set))
                print('#test:  ', len(self.test_set))
                self.num_races = kwargs['num_races']
                self.weights = [1.0] * self.num_races
            def train_dataloader(self):
                sample_weights = [self.weights[self.train_set.get_race_index(sample['race'])] for sample in self.train_set.samples]
                sampler = WeightedRandomSampler(weights=sample_weights,
                                                num_samples=len(sample_weights),
                                                replacement=True)
                return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers, sampler=sampler)
            def val_dataloader(self):
                return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
            def test_dataloader(self):
                return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
            def update_weights(self, weights):
                self.weights = weights
                
    return CheXpertDataModule(**kwargs)


def createModel(hparams, **kwargs):
    class Model(pl.LightningModule):        
        def configure_optimizers(self):
            params_to_update = []
            for param in self.parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
            optimizer = torch.optim.Adam(params_to_update, lr=0.001)
            return optimizer
        
        def validation_step(self, batch, batch_idx):
            loss = self.process_batch(batch)
            self.log('val_loss', loss)

        def test_step(self, batch, batch_idx):
            loss = self.process_batch(batch)
            self.log('test_loss', loss)

        #### TASK SPECIFIC FUNCTIONS ####

        if hparams.mode == 'baseline':
            def forward(self, x):
                x = self.model(x)
                return x
            
            def process_batch(self, batch):
                data, lab = self.unpack_batch(batch)
                out = self.forward(data)
                prob = torch.sigmoid(out)
                loss = F.binary_cross_entropy(prob, lab)
                return loss
            
            if hparams.extractor == 'densenet':
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_classes = kwargs['num_classes']
                    self.model = models.densenet121(pretrained=True)
                    num_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(num_features, self.num_classes)
                
                def remove_head(self):
                    num_features = self.model.classifier.in_features
                    id_layer = nn.Identity(num_features)
                    self.model.classifier = id_layer
                
                def unpack_batch(self, batch):
                    return batch['image'], batch['label']
                
                def training_step(self, batch, batch_idx):
                    loss = self.process_batch(batch)
                    self.log('train_loss', loss)
                    grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
                    self.logger.experiment.add_image('images', grid, self.global_step)
                    return loss
                
            elif hparams.extractor == 'foundation':
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_classes = kwargs['num_classes']
                    self.model = nn.Sequential(
                        nn.Linear(1376, self.num_classes)
                    )

                def unpack_batch(self, batch):
                    return batch['features'], batch['label'], batch['race']

                def training_step(self, batch, batch_idx):
                    loss = self.process_batch(batch)
                    self.log('train_loss', loss)
                    return loss
                
                def process_batch(self, batch):
                    feat, lab, race = self.unpack_batch(batch)
                    out = self.forward(feat, race)
                    prob = torch.sigmoid(out)
                    loss = F.binary_cross_entropy(prob, lab)
                    return loss

            else:
                raise NotImplementedError

        elif hparams.mode == 'last':
            def process_batch(self, batch):
                img, lab, race = self.unpack_batch(batch)
                out = self.forward(img, race)
                prob = torch.sigmoid(out)
                loss = F.binary_cross_entropy(prob, lab)
                return loss
            
            def forward(self, x, race):
                    x = self.model.forward(x)
                    outputs = []
                    for i, r in enumerate(race):
                        output = self.classifiers[str(r.item())](x[i])
                        outputs.append(output)
                    x = torch.stack(outputs)
                    return x
            
            if hparams.extractor == 'densenet':
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_classes = kwargs['num_classes']
                    self.num_races = kwargs['num_races']
                    self.model = models.densenet121(pretrained=True)
                    num_features = self.model.classifier.in_features
                    self.model.classifier = nn.Identity()
                    self.classifiers = nn.ModuleDict({
                        str(i): nn.Linear(num_features, self.num_classes) for i in range(self.num_races)
                    })
                
                def remove_head(self):
                    self.classifiers = nn.ModuleDict({
                        str(i): nn.Identity() for i in range(len(self.classifiers))
                    })
                
                def unpack_batch(self, batch):
                    return batch['image'], batch['label'], batch['race']
                
                def training_step(self, batch, batch_idx):
                    loss = self.process_batch(batch)
                    self.log('train_loss', loss)
                    grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
                    self.logger.experiment.add_image('images', grid, self.global_step)
                    return loss
            
            elif hparams.extractor == 'foundation':
                def __init__(self, **kwargs):
                    super().__init__()
                    self.num_classes = kwargs['num_classes']
                    self.num_races = kwargs['num_races']
                    self.model = nn.Identity()
                    self.classifiers = nn.ModuleDict({
                        str(i): nn.Linear(1376, self.num_classes) for i in range(self.num_races)
                    })

                def unpack_batch(self, batch):
                    return batch['features'], batch['label'], batch['race']

                def training_step(self, batch, batch_idx):
                    loss = self.process_batch(batch)
                    self.log('train_loss', loss)
                    return loss

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
    
    return Model


def test(model, data_loader, device, hparams, **kwargs):
    if hparams.mode == 'baseline':
        if hparams.extractor == 'densenet':
            num_classes = kwargs['num_classes']
            model.eval()
            logits = []
            preds = []
            targets = []
            with torch.no_grad():
                for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                    img, lab = batch['image'].to(device), batch['label'].to(device)
                    out = model(img)
                    pred = torch.sigmoid(out)
                    logits.append(out)
                    preds.append(pred)
                    targets.append(lab)
                logits = torch.cat(logits, dim=0)
                preds = torch.cat(preds, dim=0)
                targets = torch.cat(targets, dim=0)
                counts = []
                for i in range(0,num_classes):
                    t = targets[:, i] == 1
                    c = torch.sum(t)
                    counts.append(c)
                print(counts)
            return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()
        
        elif hparams.extractor == 'foundation':
            num_classes = kwargs['num_classes']
            model.eval()
            logits = []
            preds = []
            targets = []
            with torch.no_grad():
                for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                    feat, lab = batch['features'].to(device), batch['label'].to(device)
                    out = model(feat)
                    pred = torch.sigmoid(out)
                    logits.append(out)
                    preds.append(pred)
                    targets.append(lab)
                logits = torch.cat(logits, dim=0)
                preds = torch.cat(preds, dim=0)
                targets = torch.cat(targets, dim=0)
                counts = []
                for i in range(0,num_classes):
                    t = targets[:, i] == 1
                    c = torch.sum(t)
                    counts.append(c)
                print(counts)
            return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()
        
        else:
            raise NotImplementedError
        
    elif hparams.mode == 'last':
        if hparams.extractor == 'densenet':
            num_classes = kwargs['num_classes']
            model.eval()
            logits = []
            preds = []
            targets = []
            with torch.no_grad():
                for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                    img, lab, race = batch['image'].to(device), batch['label'].to(device), batch['race'].to(device)
                    out = model(img, race)
                    pred = torch.sigmoid(out)
                    logits.append(out)
                    preds.append(pred)
                    targets.append(lab)
                logits = torch.cat(logits, dim=0)
                preds = torch.cat(preds, dim=0)
                targets = torch.cat(targets, dim=0)
                counts = []
                for i in range(0,num_classes):
                    t = targets[:, i] == 1
                    c = torch.sum(t)
                    counts.append(c)
                print(counts)
            return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()
        
        if hparams.extractor == 'foundation':
            num_classes = kwargs['num_classes']
            model.eval()
            logits = []
            preds = []
            targets = []
            with torch.no_grad():
                for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                    feat, lab, race = batch['features'].to(device), batch['label'].to(device), batch['race'].to(device)
                    out = model(feat, race)
                    pred = torch.sigmoid(out)
                    logits.append(out)
                    preds.append(pred)
                    targets.append(lab)
                logits = torch.cat(logits, dim=0)
                preds = torch.cat(preds, dim=0)
                targets = torch.cat(targets, dim=0)
                counts = []
                for i in range(0,num_classes):
                    t = targets[:, i] == 1
                    c = torch.sum(t)
                    counts.append(c)
                print(counts)
            return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()

    else:
        raise NotImplementedError
    

def embeddings(model, data_loader, device, hparams):
    if hparams.mode == 'baseline':
        model.eval()
        embeds = []
        targets = []
        with torch.no_grad():
            for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                img, lab = batch['image'].to(device), batch['label'].to(device)
                emb = model(img)
                embeds.append(emb)
                targets.append(lab)
            embeds = torch.cat(embeds, dim=0)
            targets = torch.cat(targets, dim=0)
        return embeds.cpu().numpy(), targets.cpu().numpy()
    
    elif hparams.mode == 'last':
        model.eval()
        embeds = []
        targets = []
        with torch.no_grad():
            for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
                img, lab, race = batch['image'].to(device), batch['label'].to(device), batch['race'].to(device)
                emb = model(img, race)
                embeds.append(emb)
                targets.append(lab)
            embeds = torch.cat(embeds, dim=0)
            targets = torch.cat(targets, dim=0)
        return embeds.cpu().numpy(), targets.cpu().numpy()

    else:
        raise NotImplementedError


class WeightUpdateCallback(pl.Callback):
    def __init__(self, data_module, model, device, hparams):
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.num_races = data_module.num_races
        self.device = device
        self.hparams = hparams

    def compute_auc_per_race(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_races = []

        with torch.no_grad():
            for batch in dataloader:
                if self.hparams.extractor == "foundation":
                    inputs, labels, races = batch['features'].to(self.device), batch['label'].to(self.device), batch['race'].to(self.device)
                elif self.hparams.extractor == "densenet":
                    inputs, labels, races = batch['image'].to(self.device), batch['label'].to(self.device), batch['race'].to(self.device)
                outputs = self.model(inputs, races)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_races.append(races.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_races = np.concatenate(all_races, axis=0)

        auc_per_race = {}
        for race in range(self.num_races):
            race_indices = np.where(all_races[:, 0] == race)[0]
            race_labels = all_labels[race_indices]
            race_preds = all_preds[race_indices]
            auc = roc_auc_score(race_labels, race_preds)
            auc_per_race[race] = auc
        
        return auc_per_race
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        val_dataloader = self.data_module.val_dataloader()
        auc_per_race = self.compute_auc_per_race(val_dataloader)

        new_weights = []
        for race in range(self.num_races):
            new_weight = 1.0 / auc_per_race[race]
            new_weights.append(new_weight)
        new_weights /= sum(new_weights)

        self.data_module.update_weights(new_weights)