import os
import argparse
import SimpleITK as sitk

parser = argparse.ArgumentParser(description="Construct 3D CT volume from DICOM slices")
parser.add_argument("--dicom_dir", required=True, help="Path to folder containing DICOM .dcm slices")
parser.add_argument("--case_id", required=True, help="Case ID used in output filename")
args = parser.parse_args()

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(args.dicom_dir)
if not dicom_names:
    raise ValueError(f"No DICOM files found in {args.dicom_dir}")
reader.SetFileNames(dicom_names)
img = reader.Execute()

output_path = os.path.join(output_dir, f"{args.case_id}_ct_volume.nii.gz")
sitk.WriteImage(img, output_path)

print(f"  Size: {img.GetSize()}  Spacing: {[round(float(s), 3) for s in img.GetSpacing()]}")
print(f"  Saved -> {output_path}")