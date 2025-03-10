import nibabel as nib
from dipy.align.reslice import reslice
import os
from pathlib import Path
import numpy as np

class Resampler:
    # def __init__(self): 
    #     pass

    @staticmethod
    # Cubic spline interpolation (order=3) is suitable for MRI images, but it will blur the segmentation mask and cause non-binarization problems.
    def resample_down_to_1mm(image_filename, output_dir, output_filename, new_spacing=(1, 1, 1), mask=False):
        # if output_path is None:
        #     output_path = image_path
        if mask:
            order = 0 # Neighbor interpolation
        else:
            order = 3 # Cubic spline interpolation
        os.makedirs(output_dir, exist_ok=True)
        im = nib.load(image_filename)
        # get necessary info.
        header = im.header
        vox_zooms = header.get_zooms()
        vox_arr = im.get_fdata()
        vox_affine = im.affine
        # resample using DIPY.ALIGN
        
        # The following two lines are useless 
        # if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        #     new_spacing = (new_spacing[0], new_spacing[1], new_spacing[2])

        print(mask, new_spacing, order)
        new_vox_arr, new_vox_affine = reslice(vox_arr, vox_affine, vox_zooms, new_spacing, order=order)
        # create reoriented NIB image
        if mask:
            new_vox_arr[new_vox_arr >= 0.5] = 1
            new_vox_arr[new_vox_arr < 0.5] = 0
            new_vox_arr = new_vox_arr.astype(np.uint8)  # make sure the data type is integre, either 0 or 1
        new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)
        nib.save(new_im, os.path.join(output_dir, output_filename))
        print('Image \"' + image_filename + '\" resampled and saved to \"' + output_dir + output_filename + '\".')

if __name__ == "__main__":
    resampler = Resampler()

    # Downsample MRI data from 0.5mm to 1mm
    org_data_path = 'data_org/'
    downsampled_data_path = 'data_org_downsampled_1mm'
    mri_files = sorted([os.path.join(org_data_path, f) for f in os.listdir(org_data_path) if f.endswith(".nii.gz") or f.endswith(".nii")])
    print(f"What the MRI files need to be downsampled: {mri_files}")
    for mri_file in mri_files:
        mri_basename = os.path.splitext(mri_file)[0]
        mri_basename = Path(mri_basename).stem
        print(f"\nProcessing {mri_basename}")
        resampler.resample_down_to_1mm(mri_file, downsampled_data_path, mri_basename + '_downsampled.nii.gz', new_spacing=(1, 1, 1), mask=False)
    
    print("\nCompleted MRI data downsampling\n")

    # Downsample labels from 0.5mm to 1mm
    org_mask_data = 'labels_org/'
    org_mask_data_downsapled = 'labels_org_downsampled_1mm/'
    mask_files = sorted([os.path.join(org_mask_data, f) for f in os.listdir(org_mask_data) if f.endswith(".nii.gz") or f.endswith(".nii")])
    print(f"What the masks files need to be downsampled: {mask_files}")
    for mask_file in mask_files:
        mask_basename = os.path.splitext(mask_file)[0]
        mask_basename = Path(mask_basename).stem
        print(f"\nProcessing {mask_basename}")
        resampler.resample_down_to_1mm(mask_file, org_mask_data_downsapled, mask_basename + '_resampled.nii.gz', new_spacing=(1, 1, 1), mask=True)

    print("\nCompleted MRI mask downsampling\n")