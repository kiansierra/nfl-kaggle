#%%
import pandas as pd 
import numpy as np
import cv2
from pytorch_lightning.core import datamodule
from torch.utils.data import  DataLoader, Dataset
import pytorch_lightning as pl 
import time
import os
from PIL import Image
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
        super(VideoDataset,self).__init__()
        self.frames_df = labels_df[['video_path', 'frame']].groupby(['video_path', 'frame']).count().reset_index()
        self.labels_df = labels_df
        self.current_video_path = None
        self.current_buffer = None
        self.channels_first = True
        self.dtype = 'float16'
        self.output_channels = 1 
        self.output_size = (640,360)
        self.loading=False
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
        if video_path != self.current_video_path and not self.loading:
            self.loading = True
            self.current_video_path = video_path
            del self.current_buffer
            self.current_buffer = load_buffer(video_path, dtype=self.dtype)
            self.loading = False
        while self.loading:
            time.sleep(0.1)
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
class NFLVideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir : str, batch_size : int = 10) -> None:
        super(NFLVideoDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    def prepare_data(self, *args, **kwargs):
        train_labels_df = pd.read_csv(os.path.join(self.data_dir,'train_labels.csv'))
        test_labels_df = pd.read_csv(os.path.join(self.data_dir,'sample_submission.csv'))
        train_labels_df['video_path'] = train_labels_df['video'].apply(lambda x: os.path.join(self.data_dir, 'train', x))  
        test_labels_df['video_path'] = test_labels_df['video'].apply(lambda x: os.path.join(self.data_dir, 'test', x))    
        train_videos = train_labels_df['video'].unique()[:100]
        val_videos = train_labels_df['video'].unique()[100:]
        self.train_labels_df = train_labels_df[train_labels_df['video'].isin(train_videos)].reset_index()
        self.val_labels_df = train_labels_df[train_labels_df['video'].isin(val_videos)].reset_index()
        self.test_labels_df = test_labels_df
    def train_dataloader(self) -> DataLoader:
        dataset = VideoDataset(self.train_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size)
    def val_dataloader(self) -> DataLoader:
        dataset = VideoDataset(self.val_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size)
    def test_dataloader(self) -> DataLoader:
        dataset = VideoDataset(self.test_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size)
# %%
class ImageDataset():
    def __init__(self, labels_df : pd.DataFrame) -> None:
        #super(ImageDataset,self).__init__()
        self.frames_df = labels_df[['video', 'frame', 'image_path']].groupby(['video', 'frame', 'image_path']).count().reset_index()
        self.labels_df = labels_df
        self.current_video_path = None
        self.current_buffer = None
        self.channels_first = True
        self.dtype = 'float16'
        self.output_channels = 1 
        self.output_size = (640,360)
        self.loading=False
    def get_img_mask(self, video, frame, img_shape):
        video_cond = self.labels_df['video'] == video
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
        video, frame, img_path = row['video'], row['frame'], row['image_path']
        out_frame = cv2.imread(img_path)
        target = self.get_img_mask(video, frame, out_frame.shape)
        out_frame = cv2.resize(out_frame.astype('float32'), dsize=self.output_size, interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target.astype('float32'), dsize=self.output_size, interpolation=cv2.INTER_CUBIC)
        if len(target.shape) < 3:
            target = np.expand_dims(target,axis=-1)
        if self.channels_first: 
            out_frame = np.transpose(out_frame, (2,0,1))
            target = np.transpose(target, (2,0,1))
        return  out_frame.astype(self.dtype), target.astype(self.dtype)
# %%
class NFLImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir : str, batch_size : int = 10, num_workers=0) -> None:
        super(NFLImageDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers= num_workers
        self.num_vids_train = 5
        self.num_videos_val = 2
    def prepare_data(self, *args, **kwargs):
        train_labels_df = pd.read_csv(os.path.join(self.data_dir,'train_labels.csv'))
        test_labels_df = pd.read_csv(os.path.join(self.data_dir,'sample_submission.csv'))
        train_labels_df['start_frame'] = train_labels_df[['video', 'frame']].groupby('video').transform(min)
        test_labels_df['start_frame'] = test_labels_df[['video', 'frame']].groupby('video').transform(min)
        get_image_name = lambda x: str(max(x['frame'] -1,0))+'_' +  x['video'].split('.')[0] +'.jpeg'
        get_image_path_train = lambda x: os.path.join(self.data_dir, 'train_image', get_image_name(x))
        get_image_path_test = lambda x: os.path.join(self.data_dir, 'test_image', get_image_name(x))
        train_labels_df['image_path'] = train_labels_df[['video', 'frame', 'start_frame']].apply(get_image_path_train, axis=1)  
        test_labels_df['image_path'] = test_labels_df[['video', 'frame', 'start_frame']].apply(get_image_path_test, axis=1)   
        train_videos = train_labels_df['video'].unique()[:self.num_vids_train]
        val_videos = train_labels_df['video'].unique()[-self.num_videos_val:]
        self.train_labels_df = train_labels_df[train_labels_df['video'].isin(train_videos)].reset_index()
        self.val_labels_df = train_labels_df[train_labels_df['video'].isin(val_videos)].reset_index()
        self.test_labels_df = test_labels_df
    def train_dataloader(self) -> DataLoader:
        dataset = ImageDataset(self.train_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self) -> DataLoader:
        dataset = ImageDataset(self.val_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self) -> DataLoader:
        dataset = ImageDataset(self.test_labels_df)
        return DataLoader(dataset, batch_size = self.batch_size, num_workers=self.num_workers)        
# %%
