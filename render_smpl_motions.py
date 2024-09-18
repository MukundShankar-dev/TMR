import numpy as np
import joblib
from IPython.display import Video
import genmotion.render.python.rendermotion as rendermotion

smpl_data = joblib.load('new_smpl_files/subset_smpl_param/M001P001A001R002.pkl')
data = np.array(smpl_data['poses'])
rendermotion.render(data, "./renderings")
