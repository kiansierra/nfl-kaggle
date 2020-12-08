#%%
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import plotly.express as px
from PIL import Image, ImageDraw
import IPython.display as di
import cv2
import os
from eda_utils import *
# %%
DATA_DIR = '../nfl-impact-detection'
TRAIN_VIDEO_DIR = os.path.join(DATA_DIR, 'train')
#%%
os.listdir(DATA_DIR)
#%%
sample_sub = pd.read_csv(os.path.join(DATA_DIR,'sample_submission.csv'))
sample_sub.info()
sample_sub.head()
# %%
train_players_df = pd.read_csv(os.path.join(DATA_DIR,'train_player_tracking.csv'))
train_players_df['time'] = pd.to_datetime(train_players_df['time'])
train_players_df.info()
# %%
train_labels_df = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))
train_labels_df.info()
train_labels_df.head()
# %%
video_num = 3
video_name = os.listdir(TRAIN_VIDEO_DIR)[video_num]
video_path = os.path.join(TRAIN_VIDEO_DIR,video_name)
#%%
video_df = train_labels_df[train_labels_df['video'] == video_name]
#%%
# show_video(video_path, video_df)

# %%
train_players_df.describe()
# %%
game_key = 57583
game_df = train_players_df[train_players_df['gameKey'] == game_key].sort_values(by=['time', 'player'])
game_df['home_team'] = game_df['player'].apply(lambda x: 1 if 'H' in x else 0)
game_df['seconds']  = (game_df['time'] - game_df['time'].min()).apply(lambda x : x.total_seconds())
#%%
fig = plot_game(game_df)
fig.show(renderer='notebook')
#%%
cap = cv2.VideoCapture(video_path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()
#%%
def load_buffer(video_path):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc, ret = 0, True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf
#%%
cv2.namedWindow('frame 10')
cv2.imshow('frame 10', buf[26])
cv2.waitKey(0)

# %%
buf.nbytes // 1024 //1024
# %%
buf.dtype

# %%
buf.shape
# %%
train_labels_df['impact'].unique()
# %%
train_labels_df['impactType'].unique()
# %%
train_labels_df['confidence'].value_counts(dropna=False)
# %%
train_labels_df['visibility'].hist()
# %%
