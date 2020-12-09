#%%
import pandas as pd 
import numpy as np
import cv2
from pytorch_lightning.core import datamodule
from torch.utils.data import  DataLoader, Dataset
import pytorch_lightning as pl 
#%%
def load_buffer(video_path, dtype='float32'):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), dtype=dtype)
    fc, ret = 0, True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf
# %%
class VideoDataset(Dataset):
    def __init__(self, labels_df : pd.DataFrame) -> None:
        self.frames_df = labels_df[['video_path', 'frame']].groupby(['video_path', 'frame']).count().reset_index()
        self.labels_df = labels_df
        self.current_video_path = None
        self.current_buffer = None
        self.channels_first = True
        self.dtype = 'float16'
        self.output_channels = 1 
        self.output_size = (640,360)
    def get_target(self, video_path, frame, img_shape):
        video_cond = self.labels_df['video_path'] == video_path
        frame_cond = self.labels_df['frame'] == frame
        target_df =self.labels_df.loc[video_cond & frame_cond,:]
        out = np.zeros(shape=img_shape[:2] + (self.output_channels, ), dtype=self.dtype)
        for _,row in target_df.iterrows():
            w, h = row['width'], row['height']
            l, t = row['left'], row['top']
            out[t:t+h, l:l+w,:] = 1
        return out
    def __len__(self):
        return len(self.frames_df)
    def __getitem__(self, idx):
        row = self.frames_df.loc[idx, :]
        video_path, frame = row['video_path'], row['frame']
        if video_path != self.current_video_path:
            del self.current_buffer
            self.current_buffer = load_buffer(video_path, dtype=self.dtype)
            self.current_video_path = video_path
        out_frame = self.current_buffer[frame-1]/255
        target = self.get_target(video_path, frame, out_frame.shape)
        out_frame = cv2.resize(out_frame.astype('float32'), dsize=self.output_size, interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target.astype('float32'), dsize=self.output_size, interpolation=cv2.INTER_CUBIC)
        if len(target.shape) < 3:
            target = np.expand_dims(target,axis=-1)
        if self.channels_first: 
            out_frame = np.transpose(out_frame, (2,0,1))
            target = np.transpose(target, (2,0,1))
        return  out_frame.astype(self.dtype), target.astype(self.dtype)
# %%
class NFLDataModule(pl.LightningDataModule):
    def __init__(self, train_labels_df : pd.DataFrame, val_labels_df : pd.DataFrame, batch_size : int = 10) -> None:
        super(NFLDataModule, self).__init__()
        self.train_labels_df = train_labels_df
        self.val_labels_df = val_labels_df
        self.batch_size = batch_size
    def train_dataloader(self) -> DataLoader:
        train_dataset = VideoDataset(self.train_labels_df)
        return DataLoader(train_dataset, batch_size = self.batch_size)
    def val_dataloader(self) -> DataLoader:
        val_dataset = VideoDataset(self.val_labels_df)
        return DataLoader(val_dataset, batch_size = self.batch_size)
# %%
