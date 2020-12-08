#%%
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from PIL import Image
import cv2

def make_timestep_figure(time_df, start_time, game_key=''):
    fig,axs = plt.subplots(1, figsize=(10,5))
    time_step = time_df['time'].mean()
    axs.set_title(f"Game: {game_key} -- sec:{(time_step - start_time).seconds}")
    axs.set_xlim(left=0, right=120)
    axs.set_ylim(bottom=0, top=54)
    axs.set_facecolor('green')
    axs.scatter(time_df['x'], time_df['y'], c=time_df['home_team'])
    return fig

def make_gif(game_df, out_name = 'out_game.gif'):
    images = []
    start_time = game_df['time'].unique()[0]
    for time_step in game_df['time'].unique():
        time_df = game_df[game_df['time'] == time_step]
        fig = make_timestep_figure(time_df, start_time)
        #width, height = fig.get_size_inches() * fig.get_dpi()
        fig_shape = fig.canvas.get_width_height()[::-1] + (3,)
        canvas = FigureCanvas(fig)
        canvas.draw() 
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig_shape)
        image = Image.fromarray(img_array)
        images.append(image)
        plt.close()
    images[0].save(out_name, save_all=True, append_images=images[1:])

def add_frame_labels(frame :np.ndarray, coords : dict, text: str =''):
    start_point = (coords['left'] , coords['top'] )
    end_point = (coords['left'] + coords['width'] , coords['top'] + coords['height'])
    cv2.rectangle(frame, start_point, end_point, (255,255,255), -1)

def show_video(video_path, video_df=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    print(frame.shape)
    num_frames = 1
    while(ret):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        if video_df is not None:
            for ind, row in video_df[video_df['frame'] == num_frames].iterrows():
                add_frame_labels(frame, row)
        cv2.imshow('frame',frame)
        num_frames +=1
    cv2.destroyAllWindows()

def plot_game(game_df):
    fig = px.scatter(game_df ,x='x', y='y', color='home_team', animation_frame='seconds', size='dis', size_max=20, range_x=(0,120), range_y=(0,54))
    fig.update_layout(plot_bgcolor='rgb(50,150,50)')
    return fig
    # fig.show(renderer='notebook')