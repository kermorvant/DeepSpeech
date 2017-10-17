from __future__ import absolute_import
import scipy.io.wavfile as wav

import numpy as np
from python_speech_features import mfcc
from six.moves import range
from PIL import Image, ImageOps
import math



def imageToInputVector(image, numcep, numcontext):

    # Get mfcc coefficients
    resized_height = numcep
    width,height = image.size
    resized_width = int(math.floor(width*float(resized_height)/height))
    resized_image = image.resize((resized_width,resized_height))
    features = np.asarray(resized_image.getdata()).reshape(resized_image.size)
    features = features/255.
    #print("Num mfcc",len(features))
    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]
    #print("Num mfcc after stride",len(features))
    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))
    #print("Num mfcc after context",len(features))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*numcontext+1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    # Whiten inputs (TODO: Should we whiten?)
    # Copy the strided array so that we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
    #print ("train input",len(train_inputs))
    # Return results
    return train_inputs


def imagefile_to_input_vector(image_filename, numcep, numcontext):
    r"""
    Given a image  file at ``image_filename``, calculates ``numcep``  features.
    Appends ``numcontext`` context frames to the left and right of each time step,
     and returns this data in a numpy array.
    """
    # Load image files
    image_gray = Image.open(image_filename).convert('L')
    image = ImageOps.invert(image_gray)
    return imageToInputVector(image, numcep, numcontext)
