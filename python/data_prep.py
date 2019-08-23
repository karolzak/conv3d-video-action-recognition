import random
import os
from moviepy.editor import *
import numpy as np

def random_frames(video, target_num_frames):
    num_frames = video.shape[0]
    frames_idxs = random.sample(range(0, num_frames), target_num_frames)
    frames_idxs.sort()
    return video[frames_idxs]


def random_crop(img, width=112, height=112, x=None, y=None):
    if x is None:
        x = random.randint(0, img.shape[1] - width)
    if y is None:
        y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def crop_center(img, width=112, height=112):
    x = (img.shape[1] - width) // 2
    y = (img.shape[0] - height) // 2
    img = img[y:y+height, x:x+width]
    return img


def extract_clips(video, frames_per_clip=16, step=None):
    if step is None:
        step = frames_per_clip
    total_frames = video.shape[0]
    assert frames_per_clip <= total_frames
    extracted_clips = [np.asarray(video[i:i+frames_per_clip]) for i in range(0, total_frames-frames_per_clip, step)]
    
    return np.asarray(extracted_clips, dtype=video.dtype)


def close_clip(video):
    #del video.reader
    if video is not None:
        video.close()
    del video


def calculate_mean_std(x, channels_first=False, verbose=0):
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    all_mean = []
    all_std = []    
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):
        if ndim ==5: # videos
            mean = x[:,:,:,:,c].mean()
            std = x[:,:,:,:,c].std()
        elif ndim ==4: # images rgb or grayscale
            mean = x[:,:,:,c].mean()
            std = x[:,:,:,c].std()
        if verbose:
            print("Channel %s mean before: %s" % (c, mean))   
            print("Channel %s std before: %s" % (c, std))
            
        all_mean.append(mean)
        all_std.append(std)
    
    return np.stack((all_mean, all_std))


def preprocess_input(x, mean_std, divide_std=False, channels_first=False, verbose=0):
    x = np.asarray(x, dtype=np.float32)    
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):  
        if ndim ==5: # videos
            x[:,:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,:,c] /= mean_std[1][c]
            if verbose:
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,:,c].std()))
        elif ndim ==4: # images rgb or grayscale
            x[:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,c] /= mean_std[1][c]   
            if verbose:        
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,c].std()))
            
    return x