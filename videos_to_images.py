#%%
import cv2
import os 
from dataloader import load_buffer
from PIL import Image
#%%
DATA_DIR = '../nfl-impact-detection'
# %%
def video_to_images(video_path, train_video_dir, train_image_dir):
    buf = load_buffer(os.path.join(train_video_dir, video_path), dtype='uint8')
    video_name = video_path.split('.')[0]
    for num in range(len(buf)):
        img = Image.fromarray(buf[num])
        img.save(os.path.join(train_image_dir, f"{num}_{video_name}.jpeg"))
#%%
def convert_folder_to_images(folder, data_dir=DATA_DIR):
    train_video_dir = os.path.join(data_dir, folder)
    train_image_dir = os.path.join(data_dir, f"{folder}_image")
    if not os.path.exists(train_image_dir): os.makedirs(train_image_dir)
    vid_to_img = lambda x:video_to_images(x, train_video_dir, train_image_dir)
    list(map(vid_to_img, os.listdir(train_video_dir)))
# %%
#convert_folder_to_images('train')
convert_folder_to_images('test')
# %%
