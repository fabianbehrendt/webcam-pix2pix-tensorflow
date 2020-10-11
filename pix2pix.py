import cv2
import numpy as np

import msa.utils
from msa.capturer import Capturer
from msa.predictor import Predictor
from msa.framestats import FrameStats



capture = None # msa.capturer.Capturer, video capture wrapper
predictor = None # msa.predictor.Predictor, model for prediction

img_cap = np.empty([]) # captured image before processing
img_in = np.empty([]) # processed capture image
img_out = np.empty([]) # output from prediction model

# load predictor model
predictor = Predictor(json_path = './models/gart_canny_256.json')

img_cap = np.random(800, 600)

# interpolate (temporal blur) on input image
img_in = msa.utils.np_lerp( img_in, img_cap, 1 - params.Prediction.pre_time_lerp)

# run prediction
img_predicted = predictor.predict(img_in)[0]

# interpolate (temporal blur) on output image
img_out = msa.utils.np_lerp(img_out, img_predicted, 1 - params.Prediction.post_time_lerp)