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
#%%
DATA_DIR = '../nfl-impact-detection'
#%%

#%%
# vid_datamodule.prepare_data()
# gen = iter(vid_datamodule.train_dataloader())
# #%%
# x, y = next(gen)
# x.shape, y.shape
# #%%

# x_arr = x.permute(0,2,3,1).cpu().numpy().astype('float32')
# y_arr = y.permute(0,2,3,1).cpu().numpy().astype('float32')
# #%%
# x_arr.dtype, x_arr.shape
# #%%
# fig, axs = plt.subplots(len(x_arr), 2)
# for num, (img, mask) in enumerate(zip(x_arr, y_arr)):
#     axs[num][0].imshow(img)
#     axs[num][1].imshow(img*(1+mask)/2)

# %% 
if __name__=="__main__":
    dataloader_args = {'data_dir' : DATA_DIR,'batch_size':4 ,'num_workers':4}
    vid_datamodule = NFLImageDataModule(**dataloader_args)
    model_args ={'lr':5e-4, 'weight_decay':0.05}
    #model = UnetTransfomer(num_levels=2, **model_args)
    ckpt_path = 'logs/code-nfl-impact-detection/1jnxjrvb/checkpoints/epoch=2.ckpt'
    model = UnetTransfomer.load_from_checkpoint(ckpt_path)
    print(model)
    trainer_args = {'max_epochs' :20, 'profiler' : 'simple', 'precision' :16, 'gradient_clip_val' : 100, 'limit_val_batches' : 20, 'gpus':1 }
    # logger = TensorBoardLogger("logs", name='unet', log_graph=True)
    pl.seed_everything(42)
    wandb.login(key='355d7f0e367b84fb9f8a140be052641fbd926fb5')
    logger = WandbLogger(name='nfl', save_dir='logs',offline=False)
    logger.watch(model, log='all', log_freq=100)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    #model_chkpt = ModelCheckpoint('unet', verbose=True)
    model_chkpt = ModelCheckpoint(dirpath='unet', monitor='val_loss', filename='{epoch}-{val_loss:.2f}', verbose=True)
    trainer = pl.Trainer( logger=logger, callbacks = [ lr_monitor, model_chkpt], **trainer_args)
    trainer.fit(model, vid_datamodule)
#%%