import os
import SimpleITK as sitk


def convert_dicom_to_nifti(dicom_dir, output_path):
    target_dir = dicom_dir
    for root, dirs, files in os.walk(dicom_dir):
        if any(f.endswith(".dcm") for f in files):
            target_dir = root
            break

    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(target_dir)
    if not series_IDs:
        raise ValueError(f"No DICOM series found in {target_dir}")

    largest_series = []
    for series_id in series_IDs:
        filenames = reader.GetGDCMSeriesFileNames(target_dir, series_id)
        if len(filenames) > len(largest_series):
            largest_series = filenames

    if not largest_series:
        raise ValueError(f"No DICOM files found in {dicom_dir} or its subdirectories.")

    reader.SetFileNames(largest_series)
    img = reader.Execute()
    sitk.WriteImage(img, output_path)

    num_slices = len(largest_series)
    size = img.GetSize()
    spacing = img.GetSpacing()
    print(f"  Loaded {num_slices} slices")
    print(f"  Size: {size}  Spacing: {[round(float(s), 3) for s in spacing]}")
    return img
