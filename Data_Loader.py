import os
import torch
import numpy as np
import nibabel as nib

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, data_directory):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_directory = data_directory

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Unpack the ID to find the subject folder and timepoint
        subject_folder, timepoint = self.list_IDs[index].split('_')

        # Load filtered and unfiltered data for the specified timepoint
        filtered_path = os.path.join(self.data_directory, subject_folder, 'DCE_4D_U.nii.gz') # U = filtered
        unfiltered_path = os.path.join(self.data_directory, subject_folder, 'DCE_4D_F.nii.gz') # F = unfiltered
        
        # Load the NIfTI files and extract the specific timepoint data
        filtered_img = nib.load(filtered_path).slicer[..., int(timepoint)]
        unfiltered_img = nib.load(unfiltered_path).slicer[..., int(timepoint)]
        
        # Convert to PyTorch tensors
        X = torch.from_numpy(filtered_img.get_fdata()).float()
        Y = torch.from_numpy(unfiltered_img.get_fdata()).float()

        return X, Y, subject_folder


def prepare_data(data_folder):
    """
    Prepare identifiers for DCE MRI data for all subjects in the data folder.
    
    Parameters:
    - data_folder (str): The path to the folder containing all subject subfolders.
    
    Returns:
    - list_IDs (list): A list of identifiers corresponding to each data sample.
    """
    list_IDs = []
    
    # Loop through each subject folder in the data folder
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):  # Ensure it's a directory
            # Determine the number of time points by loading one of the files
            num_timepoints = nib.load(os.path.join(subject_path, 'DCE_4D_U.nii.gz')).shape[-1] # doesnt matter which file, bc same size
            
            # Generate identifiers for each timepoint
            for timepoint in range(num_timepoints):
                id = f'{subject_folder}_{timepoint}'
                list_IDs.append(id)
                
    return list_IDs

