import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch

if __name__ == '__main__':
    # chapman_file_path = os.path.dirname(os.getcwd()) + "\\Datasets\\Chapman"
    # path = os.path.join(chapman_file_path, "contrastive_msml\\leads_['II', 'V2', 'aVL', 'aVR']")

    # with open(os.path.join(path, 'frames_phases_chapman.pkl'), 'rb') as f:
    #         input_array = pickle.load(f)
    #     # with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
    #     #     input_array = pickle.load(f)
    # """ Dict Containing Actual Labels """
    # with open(os.path.join(path,'labels_phases_chapman.pkl'), 'rb') as g:
    #     output_array = pickle.load(g)
    # # with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
    # #     output_array = pickle.load(g)
    # """ Dict Containing Patient Numbers """
    # with open(os.path.join(path,'pid_phases_chapman.pkl'), 'rb') as h:
    #     pid_array = pickle.load(h) #needed for CPPC (ours)
    # # with open(os.path.join(path,'pid_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as h:
    # #     pid_array = pickle.load(h) #needed for CPPC (ours)

    # print(input_array)

    frame_views = torch.empty(2,7,4)
    # print(frame_views.shape)
    # print("----------------")
    # print(frame_views[0,:,0].shape)

    for n in range(2):
        """ Obtain Differing 'Views' of Same Instance by Perturbing Input Frame """
        # frame = self.obtain_perturbed_frame(input_frame)
        if n == 0:
            frame = np.ones((7,4))
        else:
            frame = np.zeros((7,4))
        """ Normalize Data Frame """
        # frame = self.normalize_frame(frame)
        frame = torch.tensor(frame,dtype=torch.float)
        # label = torch.tensor(label,dtype=torch.float)
        """ Frame Input Has 1 Channel """
        frame = frame.unsqueeze(0)
        """ Populate Frame Views """
        frame_views[n,:,:] = frame

    print(frame_views)


    
                


    
    