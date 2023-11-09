import os
import nibabel as nib

def load_and_split_data(data_folder):
    """
    Load DCE MRI data for all subjects in the data folder and split along the time series dimension.
    
    Parameters:
    - data_folder (str): The path to the folder containing all subject subfolders.
    
    Returns:
    - filtered_volumes (dict): A dictionary where keys are subject IDs and values are lists of 3D arrays, each representing a timepoint in the filtered data.
    - unfiltered_volumes (dict): A dictionary where keys are subject IDs and values are lists of 3D arrays, each representing a timepoint in the unfiltered data.
    """
    # Dictionaries to store the split volumes for each subject
    filtered_volumes = {}
    unfiltered_volumes = {}
    
    # Loop through each subject folder in the data folder
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):  # Ensure it's a directory
            # Define the file paths
            input_filename = os.path.join(subject_path, 'DCE_4D_F.nii.gz')
            output_filename = os.path.join(subject_path, 'DCE_4D_U.nii.gz')
            
            # Load the 4D data
            input_data = nib.load(input_filename).get_fdata()
            output_data = nib.load(output_filename).get_fdata()
            
            # Split along the fourth dimension (time series)
            filtered_volumes[subject_folder] = [input_data[:, :, :, i] for i in range(input_data.shape[3])]
            unfiltered_volumes[subject_folder] = [output_data[:, :, :, i] for i in range(output_data.shape[3])]
            
    return filtered_volumes, unfiltered_volumes

# Example of how to use the function:
# data_folder_path = 'Data'
# all_filtered_volumes, all_unfiltered_volumes = load_and_split_data(data_folder_path)


