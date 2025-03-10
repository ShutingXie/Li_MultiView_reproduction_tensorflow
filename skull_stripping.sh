#!/bin/bash

# -f 0.5: This controls how aggressively the algorithm removes non-brain structures. A higher value (e.g., 0.6–0.7) removes more brain tissue, including CSF. A lower value (e.g., 0.2–0.3) keeps more of the surrounding tissue, including CSF.
#-g 0: This is a bias field correction factor (default: 0). Adjusting this can help in images with intensity non-uniformity.
# -m: This generates a binary brain mask alongside the brain-extracted image.

# Define input and output directories
########################################################
# When using images with resolution 1mm
inpath=data_org_downsampled_1mm
outpath=data_org_downsampled_1mm_ss

# When using images with resolution 0.5mm
# inpath=data_org
# outpath=data_org_ss
########################################################

# Ensure the output directory exists
mkdir -p $outpath

# Set up FSL environment
FSLDIR=/Users/shutingxie/fsl/
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH

# Loop over all .nii files in the input directory
for filepath in ${inpath}/*.nii.gz; do
    # Extract filename without extension
    filename=$(basename "$filepath" .nii.gz)

    echo "Processing $filename"

    # Run bet2 for skull stripping
    # /Users/shutingxie/fsl/share/fsl/bin/bet2 "$inpath/${filename}.nii" "$outpath/${filename}_ss.nii.gz" -f 0.5 -g 0 -m
	/Users/shutingxie/fsl/share/fsl/bin/bet2 "$inpath/${filename}.nii.gz" "$outpath/${filename}_ss.nii.gz" -f 0.5 -g 0

    echo "Saved to $outpath/${filename}_ss.nii.gz"
done

echo "Processing complete!"



