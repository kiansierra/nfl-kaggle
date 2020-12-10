#%%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import cv2
import os
from dataloader import NFLVideoDataModule,NFLImageDataModule, load_buffer
import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from basemodel import UnetTransfomer
import wandb
import tqdm 
#%%
DATA_DIR = '../nfl-impact-detection'

# #%%
# train_labels_df = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))
# #%%
# train_labels_df[['video', 'frame']].groupby('video').transform(min)
#%%
dataloader_args = {'data_dir' : DATA_DIR,'batch_size':6, 'num_workers':0}
vid_datamodule = NFLImageDataModule(**dataloader_args)
#%%
vid_datamodule.prepare_data()
#%%
for batch in tqdm.tqdm(vid_datamodule.train_dataloader()):
    x, y = batch