import os
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import scipy
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

K.set_image_data_format('channels_last')
smooth = 1.0

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs)  
    relu = Activation('relu')(conv)
    return relu

## a component of U-Net, to ensure the sizes between skip connections are matched
def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.shape[2] - refer.shape[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw//2), int(cw//2) + 1
        else:
            cw1, cw2 = int(cw//2), int(cw//2)
        # height, the 2nd dimension
        ch = (target.shape[1] - refer.shape[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch//2), int(ch//2) + 1
        else:
            ch1, ch2 = int(ch//2), int(ch//2)

        return (ch1, ch2), (cw1, cw2)

def get_unet(img_shape = None, first5=False):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_bn_relu(32, filters, inputs)
    conv1 = conv_bn_relu(32, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu(64, 3, pool1)
    conv2 = conv_bn_relu(64, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn_relu(96, 3, pool2)
    conv3 = conv_bn_relu(96, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(128, 3, pool3)
    conv4 = conv_bn_relu(128, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(256, 3, pool4)
    conv5 = conv_bn_relu(256, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(128, 3, up6)
    conv6 = conv_bn_relu(128, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(96, 3, up7)
    conv7 = conv_bn_relu(96, 3, conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(64, 3, up8)
    conv8 = conv_bn_relu(64, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(32, 3, up9)
    conv9 = conv_bn_relu(32, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(learning_rate=2e-4), loss=dice_coef_loss)

    return model

## crop or pad the volumes to be a standard size given the original size
def crop_or_pad(input_array, per_, ori_size):
# dim_1 = np.shape(input_array)[0]
    dim_2 = np.shape(input_array)[1]
    dim_3 = np.shape(input_array)[2]
    rows = ori_size[1]
    cols = ori_size[2]
    array_1 = np.zeros(ori_size, dtype = 'float32')
    array_1[...] = np.min(input_array)
    
    if dim_2 <=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows - dim_2)/2):(int((rows - dim_2)/2)+ dim_2), int((cols - dim_3)/2):(int((cols - dim_3)/2)+dim_3)] = input_array[:, :, :, 0]
    elif dim_2>=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, :] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), int((dim_3-cols)/2):(int((dim_3-cols)/2)+cols), 0]
    elif dim_2>=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, int((cols-dim_3)/2):(int((cols-dim_3)/2)+dim_3)] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), :, 0]
    elif dim_2<=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows-dim_2)/2):(int((rows-dim_2)/2)+ dim_2), :] = input_array[:, :, int((dim_3 -cols)/2):(int((dim_3 -cols)/2)+cols), 0]
    return array_1

## pre-processing, crop or pad the volume with a reference size
def pre_processing(volume, per_, ref_size):
    rows, cols = ref_size[0], ref_size[1]
    dim_1 = np.shape(volume)[0]
    orig_rows, orig_cols = np.shape(volume)[1], np.shape(volume)[2]
    cropped_volume = []
    for nn in range(np.shape(volume)[0]):
        min_value = np.min(volume)
        if orig_rows >= rows and orig_cols >= cols:
            cropped_volume.append(volume[nn, int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows,
                                int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols])
        elif orig_rows >= rows and cols >= orig_cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[:, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, 
                                                    int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows, :]
            cropped_volume.append(norm_slice)
        elif rows >= orig_rows and orig_cols >= cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, :] = volume[nn, :, int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols]
            cropped_volume.append(norm_slice)
        elif rows >= orig_rows and cols >= orig_cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, :, :]
            cropped_volume.append(norm_slice)
    cropped_volume = np.asarray(cropped_volume)
    cropped_volume = cropped_volume[int(dim_1*per_): (dim_1 -int(dim_1*per_))]
    return cropped_volume[..., np.newaxis]

def inverse_orient(orient_):
    inv_orient = []
    if orient_ == [0, 1, 2]:
        inv_orient = (0, 1, 2)
    elif orient_ == [1, 0, 2]:
        inv_orient = (1, 0, 2)
    elif orient_ == [1, 2, 0]:
        inv_orient = (2, 0, 1)
    elif orient_ == [2, 1, 0]:
        inv_orient = (2, 1, 0)
    return inv_orient

## simple post-processing
def post_processing(input_array, ori_size, orient_1):
# output_array = np.zeros(ori_size, dtype= 'float32')
    output_array = crop_or_pad(input_array, per, ori_size)
    inv_orient = inverse_orient(orient_1)
    output_array = np.transpose(output_array, inv_orient)
    return output_array


def anatomically_consistent_postprocessing(input_array, ori_size, orient_1, removal_ratio=0.2):
    """
    Based on the original post_processing results, the false positives are 
    removed from the slices with the first and last removal_ratio ratios, that is, all the segmentation results in these slices are set to 0.
    """
    output_array = post_processing(input_array, ori_size, orient_1)

    num_slices = output_array.shape[0]
    m = int(removal_ratio * num_slices)

    output_array[:m, :, :] = 0
    output_array[-m:, :, :] = 0
    
    return output_array


def visualize_image_and_mask(image_volume, mask_volume, num_slices=30, output_dir='visualization_output/overlay', view='coronal'):
    """
    Randomly select a number of slices from the image_volume and its corresponding mask_volume,
    display them side by side, and save the visualizations as separate images.
    """
    # ensure output folder exists
    view_output_dir = os.path.join(output_dir, view)
    os.makedirs(view_output_dir, exist_ok=True)
    
    num_total = image_volume.shape[0]
    # select random slice indices (or all if fewer slices than num_slices)
    selected_indices = random.sample(range(num_total), min(num_slices, num_total))
    
    for idx in selected_indices:
        # Get the image and mask slice (squeeze the last dimension for plotting)
        img_slice = image_volume[idx, :, :, 0]
        mask_slice = mask_volume[idx, :, :, 0]
        
        # Create a figure with two subplots: left for image, right for mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_slice, cmap='gray')
        plt.title(f"{view.capitalize()} Image Slice {idx}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask_slice, cmap='gray')
        plt.title(f"{view.capitalize()} Mask Slice {idx}")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(view_output_dir, f"{view}_slice_{idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Available devices:", tf.config.list_physical_devices())

    direction_1 = 'coronal'
    direction_2 = 'axial'

    per = 0.2
    thresh = 10
    # threshold = 0.5
    img_shape = (180, 180, 1)
    model = get_unet(img_shape)
    model_path = 'models/'
    input_data_path = 'data_org_downsampled_1mm_ss'
    output_path = 'predictions_Li_1mm/'
    image_path = 'images'
    pred_fina_global = np.array([])
    label_array_golabel = np.array([])

    def check_model_file(filepath):
        """ check whether the h5 files exist or not"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return filepath

    os.makedirs(output_path, exist_ok=True)

    mri_files = sorted([os.path.join(input_data_path, f) for f in os.listdir(input_data_path) if f.endswith(".nii.gz") or f.endswith(".nii")])
    print(mri_files)

    # Pass through each MRI data to make predictions
    for file_index, mri_file in enumerate(mri_files):
        print(mri_file)
        
        reampled_base = os.path.basename(mri_file)
        reampled_base = os.path.splitext(reampled_base)[0]
        mri_basename = Path(reampled_base).stem
        print(f"\nProcessing {mri_basename}")

        # Read MRI images
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(mri_file))
        print(f"MRI {mri_file} shape: {image_array.shape}") # eg. (340, 488, 320) --> (170, 244, 160)

        # z-score normalization
        brain_mask_T1 = np.zeros(np.shape(image_array), dtype = 'float32')
        brain_mask_T1[image_array >=thresh] = 1
        brain_mask_T1[image_array < thresh] = 0
        for iii in range(np.shape(image_array)[0]):
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside br
        image_array = image_array - np.mean(image_array[brain_mask_T1 == 1])
        image_array /= np.std(image_array[brain_mask_T1 == 1])
            
        # Transform/project the original array to axial and coronal views
        corona_array = np.transpose(image_array, (1, 0, 2))
        print(f"Coronal shape: {corona_array.shape}")
        axial_array = image_array
        print(f"Axial shape: {axial_array.shape}")

        # This is to check the orientation of your images is right or not. Please check /images/coronal and /images/axial
        for ss in range(np.shape(corona_array)[0]):
            slice_ = 255*(corona_array[ss] - np.min(corona_array[ss]))/(np.max(corona_array[ss]) - np.min(corona_array[ss]))
            im = Image.fromarray(np.uint8(slice_))
            im.save(os.path.join(image_path, direction_1, str(ss)+'.png'))
        
        for ss in range(np.shape(axial_array)[0]):
            slice_ = 255*(axial_array[ss] - np.min(axial_array[ss]))/(np.max(axial_array[ss]) - np.min(axial_array[ss]))
            im = Image.fromarray(np.uint8(slice_))
            im.save(os.path.join(image_path, direction_2, str(ss)+'.png')) 
        
        # Original size and the orientations 
        ori_size_c = np.asarray(np.shape(corona_array))
        ori_size_a = np.asarray(np.shape(axial_array))
        orient_c = [1, 0, 2]
        orient_a = [0, 1, 2]
        
        # Pre-processing, crop or pad them to a standard size [N, 180, 180]
        corona_array =  pre_processing(corona_array, per, img_shape)
        print(f"Coronal shape: {corona_array.shape}")
        axial_array =  pre_processing(axial_array, per, img_shape)
        print(f"Axial shape: {axial_array.shape}")

        # Check the values in tensor are valid, not nan for inf
        print(f"Coronal array contains NaN: {np.isnan(corona_array).any()}")
        print(f"Coronal array contains Inf: {np.isinf(corona_array).any()}")
        print(f"Coronal array type: {type(corona_array)}")
        print(f"Coronal array data type: {corona_array.dtype}")
        print(f"Axial array contains NaN: {np.isnan(axial_array).any()}")
        print(f"Axial array contains Inf: {np.isinf(axial_array).any()}")
        print(f"Axial array type: {type(axial_array)}")
        print(f"Axial array type: {axial_array.dtype}")

        # Check model
        print("Model summary:")
        print(model)
        
        # Do inference on coronal view 
        model_path_1 = check_model_file(os.path.join(model_path,direction_1+'_0.h5'))
        model.load_weights(model_path_1)
        pred_1c = model.predict(corona_array, batch_size=1, verbose=True)

        model_path_2 = check_model_file(os.path.join(model_path,direction_1+'_1.h5'))
        model.load_weights(model_path_2)
        pred_2c = model.predict(corona_array, batch_size=1, verbose=True)

        model_path_3 = check_model_file(os.path.join(model_path,direction_1+'_2.h5'))
        model.load_weights(model_path_3)
        pred_3c = model.predict(corona_array, batch_size=1, verbose=True)

        # Ensemble 
        pred_c = (pred_1c + pred_2c + pred_3c) / 3  # Fuse multiple predictions

        # Do inference on axial view
        model_path_1 = check_model_file(os.path.join(model_path,direction_2+'_0.h5'))
        model.load_weights(model_path_1)
        pred_1a = model.predict(axial_array, batch_size=1, verbose=True)

        model_path_2 = check_model_file(os.path.join(model_path,direction_2+'_1.h5'))
        model.load_weights(model_path_2)
        pred_2a = model.predict(axial_array, batch_size=1, verbose=True)

        model_path_3 = check_model_file(os.path.join(model_path,direction_2+'_2.h5'))
        model.load_weights(model_path_3)
        pred_3a = model.predict(axial_array, batch_size=1, verbose=True)

        pred_a = (pred_1a + pred_2a + pred_3a) / 3 # Fuse multiple predictions

        # Transform them to their original size and orientations
        # pred_1_post = post_processing(pred_c, ori_size_c, orient_c)
        # pred_2_post = post_processing(pred_a, ori_size_a, orient_a)
        pred_1_post = anatomically_consistent_postprocessing(pred_c, ori_size_c, orient_c, removal_ratio=0.2)
        pred_2_post = anatomically_consistent_postprocessing(pred_a, ori_size_a, orient_a, removal_ratio=0.2)

            
        # Ensemble of two views
        ####################################################################################
        ################  Here is the prediction results for 1mm resolution ################
        pred_final = (pred_1_post + pred_2_post) / 2
        # The following two lines are provided by authors
        # pred_final[pred_final >= 0.4] = 1.
        # pred_final[pred_final < 0.4] = 0.
        
        # I rewrited the following two lines in order to make the threshold value adjustable
        # pred_final[pred_final >= threshold] = 1.
        # pred_final[pred_final < threshold] = 0.
        ####################################################################################
        ####################################################################################
        
        # Reconstruct 3D images in NIfTI format
        # Step1: read original images
        org_img = sitk.ReadImage(mri_file)

        # Step2: Convert predictions results from array to SimpleITK image
        pred_sitk = sitk.GetImageFromArray(pred_final)

        # Step3: Paste original information to prediction results
        pred_sitk.CopyInformation(org_img)
        # Or manually set up all the information
        # pred_sitk.SetOrigin(original_img.GetOrigin())
        # pred_sitk.SetSpacing(original_img.GetSpacing())
        # pred_sitk.SetDirection(original_img.GetDirection())

        # Step4: Save the prediction image to NIfTI format
        filename_resultImage = os.path.join(output_path, mri_basename + '_pred.nii.gz')
        sitk.WriteImage(pred_sitk, filename_resultImage)
        print(f"Saved result to {filename_resultImage}")

    # Double-check the threshold value
    # print(f"\nThreshold: {threshold}\n")