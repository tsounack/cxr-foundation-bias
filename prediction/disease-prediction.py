import os
import torch
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

import utils

image_size = (224, 224)
num_classes = 14
num_races = 3
batch_size = 256
num_workers = 36

img_data_dir = '/data4/CheXpert/'
data_dir = '/data4/CheXpert/chexpert_embeddings'
dict1_path = '/data4/CheXpert/train.npz'
dict2_path = '/data4/CheXpert/validation.npz'
csv_train_img='../datafiles/chexpert/chexpert.sample.train.csv'
csv_val_img='../datafiles/chexpert/chexpert.sample.val.csv'
csv_test_img='../datafiles/chexpert/chexpert.resample.test.csv'


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    CheXpertDataset = utils.createCheXpertDataset(hparams, img_data_dir, data_dir, dict1_path, dict2_path)

    train_set = CheXpertDataset(csv_train_img, image_size, augmentation=True, pseudo_rgb=True)
    val_set = CheXpertDataset(csv_val_img, image_size, augmentation=False, pseudo_rgb=True)
    test_set = CheXpertDataset(csv_test_img, image_size, augmentation=False, pseudo_rgb=True)

    for run in range(hparams.n_runs):
        print(f'Run {run}')
        # data
        data = utils.createCheXpertDataModule(hparams,
                                              image_size=image_size,
                                              pseudo_rgb=True,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              train_set=train_set,
                                              val_set=val_set,
                                              test_set=test_set,
                                              num_races=num_races)

        # model
        model_type = utils.createModel(hparams)
        model = model_type(num_classes=num_classes, num_races=num_races)

        # Create output directory
        out_name = f"{hparams.extractor}_{hparams.mode}"
        if hparams.sampler != 'none':
            out_name += f"_{hparams.sampler}"
        out_dir = './' + out_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

        # train
        if hparams.sampler == 'weighted':
            weight_update_callback = utils.WeightUpdateCallback(data, model, device, hparams)
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, weight_update_callback],
                log_every_n_steps = 5,
                max_epochs=hparams.n_epochs,
                accelerator='auto',
                devices='auto',
                logger=TensorBoardLogger('.', name=out_name)
            )
        else:
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback],
                log_every_n_steps = 5,
                max_epochs=hparams.n_epochs,
                accelerator='auto',
                devices='auto',
                logger=TensorBoardLogger('.', name=out_name)
            )

        trainer.logger._default_hp_metric = False
        trainer.fit(model, data)

        model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes, num_races=num_races)

        model.to(device)

        cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
        cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
        cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

        print('VALIDATION')
        preds_val, targets_val, logits_val = utils.test(model, data.val_dataloader(), device, hparams, num_classes=num_classes)
        df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
        df = pd.concat([df, df_logits, df_targets], axis=1)
        df.to_csv(os.path.join(out_dir, f'{run}_predictions.val.csv'), index=False)

        print('TESTING')
        preds_test, targets_test, logits_test = utils.test(model, data.test_dataloader(), device, hparams, num_classes=num_classes)
        df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df = pd.concat([df, df_logits, df_targets], axis=1)
        df.to_csv(os.path.join(out_dir, f'{run}_predictions.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dev', default=0)
    parser.add_argument('--extractor', default='densenet')
    parser.add_argument('--mode', default='baseline')
    parser.add_argument('--sampler', default='none')
    parser.add_argument('--n_runs', default=10, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    args = parser.parse_args()

    main(args)
