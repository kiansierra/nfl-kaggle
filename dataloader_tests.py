#%%
from dataloader import NFLVideoDataModule,NFLImageDataModule, load_buffer
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
    x, y, bbox = batch
    assert bbox.shape == (dataloader_args['batch_size'], 41, 4)