import random
from moviepy.editor import *
import numpy as np


def get_test_split(test_split_file='../data/ucf101/testlist03.txt'):
    """
    Loads a text file with a list of filenames that should be used as test dataset
    
    Parameters
    ----------
    test_split_file : str, optional
        File path to test split text file, by default '../data/ucf101/testlist03.txt'
    
    Returns
    -------
    list(string)
        Returns a list of filenames for test dataset
    """
    with open(test_split_file, 'r') as f:
        test_split = f.readlines()
        test_split = list(map(lambda file: file.replace('\n','').split('/')[1], test_split))
    return test_split


def get_classes(classes_file='../data/ucf101/classInd.txt'):
    """
    Loads a text file with class->id mapping
    
    Parameters
    ----------
    classes_file : str, optional
        File path to class->id mapping text file, by default '../data/ucf101/classInd.txt'
    
    Returns
    -------
    dict
        Dictionary of class names and numeral id
        Example: {'Class1': 1, 'Class2': 2}
    """
    with open(classes_file, 'r') as f:
        classes = f.readlines()
        classes = map(lambda cls: cls.replace('\n','').split(' '), classes)
        classes = dict(map(lambda cls: (cls[1], int(cls[0])), classes))
    return classes


def random_frames(video, target_num_frames):
    """
    From an input video clip it randomly selects and returns a specified number of frames in sorted order
    
    Parameters
    ----------
    video : array, required
        Array of multiple elements (frames)
    target_num_frames : int, required
        How many elements (frames) to randomly select and return
    
    Returns
    -------
    array
        Subsample of input array containing randomly selected elements (frames)
    """
    num_frames = video.shape[0]
    frames_idxs = random.sample(range(0, num_frames), target_num_frames)
    frames_idxs.sort()
    return video[frames_idxs]


def crop_frame(img, width, height, x, y):
    """
    Returns a crop of image (frame) based on specified parameters
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, required
        Width of the crop
    height : int, required
        Height of the crop
    x : int, required
        X position of the crop, by default None
    y : int, required
        Y position of the crop, by default None
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    img = img[y:y+height, x:x+width]
    return img

def random_crop(img, width=112, height=112):
    """
    Returns a random crop of image (frame) based on specified width and height
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, optional
        Width of the crop, by default 112
    height : int, optional
        Height of the crop, by default 112
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    return crop_frame(img, width, height, x, y)


def crop_center(img, width=112, height=112):
    """
    Returns a centered crop of image (frame) based on specified width and height
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, optional
        Width of the crop, by default 112
    height : int, optional
        Height of the crop, by default 112
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    x = (img.shape[1] - width) // 2
    y = (img.shape[0] - height) // 2
    return crop_frame(img, width, height, x, y)


def extract_clips(video, frames_per_clip=16, step=None):
    """
    Extracts clips from input clip based on specified parameters
    
    Parameters
    ----------
    video : array
        Array representing a video consisting of multiple frames (images)
    frames_per_clip : int, optional
        Number of frames for output clips, by default 16
    step : int, optional
        Step value to iterate through input frames, by default None
    
    Returns
    -------
    array
        Array of video clips extracted from input video clip
    """
    if step is None:
        step = frames_per_clip
    total_frames = video.shape[0]
    assert frames_per_clip <= total_frames
    extracted_clips = [np.asarray(video[i:i+frames_per_clip]) for i in range(0, total_frames-frames_per_clip, step)]
    
    return np.asarray(extracted_clips, dtype=video.dtype)


def close_clip(video):
    """
    Closes the connection to the video file
    
    Parameters
    ----------
    video : VideoFileClip object
        MoviePy VideoFileClip object to close and delete
    """
    if video is not None:
        video.close()
    del video


def calculate_mean_std(x, channels_first=False, verbose=0):
    """
    Calculates channel-wise mean and std
    
    Parameters
    ----------
    x : array
        Array representing a collection of images (frames) or
        collection of collections of images (frames) - namely video
    channels_first : bool, optional
        Leave False, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array of shape [2, num_channels]
        Array with per channel mean and std for all the frames
    """
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
    """
    Channel-wise substraction of mean from the input and optional division by std
    
    Parameters
    ----------
    x : array
        Input array of images (frames) or videos
    mean_std : array
        Array of shape [2, num_channels] with per-channel mean and std
    divide_std : bool, optional
        Add division by std or not, by default False
    channels_first : bool, optional
        Leave False, otherwise not implemented, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array
        Returns input array after applying preprocessing steps
    """
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


def predict_c3d(x, model):
    """
    Runs predictions on specified model and returns them
    
    Parameters
    ----------
    x : array
        Input array with data propper for input shape of the model
    model : Keras model object
        Model object that will be used for inferencing
    
    Returns
    -------
    array
        Array with output predictions returned by Keras model
    """
    pred = []
    for batch in x:
        pred.append(model.predict(batch))
    return pred