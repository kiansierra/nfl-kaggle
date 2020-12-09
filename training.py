#%%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import cv2
import os
from dataloader import NFLDataModule, load_buffer
import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from basemodel import UnetTransfomer
import wandb
DATA_DIR = '../nfl-impact-detection'
TRAIN_VIDEO_DIR = os.path.join(DATA_DIR, 'train')
#%%
TRAIN_ARRAY_DIR = os.path.join(DATA_DIR, 'train_array')
# %%
train_labels_df = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))
#%%
train_labels_df['video_path'] = train_labels_df['video'].apply(lambda x: os.path.join(TRAIN_VIDEO_DIR, x))     
#%%
video_path = train_labels_df['video_path'].unique()[0]
video_path
#%%
# buf = load_buffer(video_path)
# #%%
# buf.shape
# %%time
# buf = load_buffer(video_path)
# #%%
# np.save('video.npy', buf)
# #%%
# %%time
# vid = np.load('video.npy')
#%%
train_videos = train_labels_df['video'].unique()[:100]
val_videos = train_labels_df['video'].unique()[100:]
#%%
train_df = train_labels_df[train_labels_df['video'].isin(train_videos)].reset_index()
val_df = train_labels_df[train_labels_df['video'].isin(val_videos)].reset_index()
#%%
vid_datamodule = NFLDataModule(train_df, val_df, batch_size=2)
#%%
model = UnetTransfomer(num_levels=2)
print(model)
#%%
trainer_args = {'max_epochs' :20, 'profiler' : 'simple', 'precision' :16, 'gradient_clip_val' : 100, 'limit_val_batches' : 20, 'gpus':1 }
#logger = TensorBoardLogger("logs", name='unet', log_graph=True)
pl.seed_everything(42)
wandb.login(key='355d7f0e367b84fb9f8a140be052641fbd926fb5')
logger = WandbLogger(name='nfl', save_dir='logs',offline=True)
logger.watch(model, log='gradients', log_freq=100)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_chkpt = ModelCheckpoint('unet', verbose=True)
trainer = pl.Trainer( logger=logger, callbacks = [ lr_monitor, model_chkpt], **trainer_args)
# %%
trainer.fit(model, vid_datamodule)
#%%