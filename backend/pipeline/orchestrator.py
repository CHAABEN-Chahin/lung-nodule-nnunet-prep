import os
import sys
import tempfile
import shutil
import traceback
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, CLASSIFIER_PATH, TEMP_DIR

from pipeline.construct_3d import convert_dicom_to_nifti
from pipeline.lung_mask import extract_lung_and_lobe_masks
from pipeline.predict import create_patches_and_infer
from pipeline.filter_nodules import filter_false_positives
from pipeline.extract_features import extract_all_features
from pipeline.generate_report import generate_report


pipeline_progress = {}


def get_progress(case_id):
    return pipeline_progress.get(case_id)


def run_pipeline(case_id, dicom_path, case_dir, db_path, patient_info):
    """
    Full pipeline: DICOM -> 3D Volume -> Lung/Lobe Masks -> nnU-Net ->
    False Positive Filter -> Feature Extraction -> Report.
    """
    import sqlite3

    def update_db(status, num_nodules=0, error=None):
        db = sqlite3.connect(db_path)
        if error:
            db.execute(
                "UPDATE cases SET status=?, error_message=? WHERE case_id=?",
                (status, error, case_id),
            )
        else:
            db.execute(
                "UPDATE cases SET status=?, num_nodules=? WHERE case_id=?",
                (status, num_nodules, case_id),
            )
        db.commit()
        db.close()

    def set_progress(phase, total_phases, message):
        pipeline_progress[case_id] = {
            "phase": phase,
            "total_phases": total_phases,
            "message": message,
        }
        print(f"[{case_id}] Phase {phase}/{total_phases}: {message}")

    os.makedirs(case_dir, exist_ok=True)
    nodules_dir = os.path.join(case_dir, "nodules")
    os.makedirs(nodules_dir, exist_ok=True)

    ct_path = os.path.join(case_dir, "ct_volume.nii.gz")
    lung_mask_path = os.path.join(case_dir, "lung_mask.nii.gz")
    lobe_mask_path = os.path.join(case_dir, "lobe_masks.nii.gz")
    nodule_mask_raw_path = os.path.join(case_dir, "nodule_mask_raw.nii.gz")
    nodule_mask_path = os.path.join(case_dir, "nodule_mask.nii.gz")
    summary_path = os.path.join(nodules_dir, "summary.json")
    report_path = os.path.join(case_dir, "report.pdf")

    total_phases = 7
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)

    try:
        update_db("processing")

        # Phase 1: DICOM -> NIfTI
        set_progress(1, total_phases, "Converting DICOM to 3D volume")
        convert_dicom_to_nifti(dicom_path, ct_path)

        # Phase 2: Lung + Lobe mask extraction
        set_progress(2, total_phases, "Extracting lung and lobe masks (TotalSegmentator)")
        lung_voxels = extract_lung_and_lobe_masks(ct_path, lung_mask_path, lobe_mask_path)
        if lung_voxels == 0:
            update_db("complete", 0)
            set_progress(total_phases, total_phases, "No lungs detected - empty result")
            _write_empty_summary(summary_path)
            return

        # Phase 3+4+5: Patch creation + nnU-Net inference + stitching
        set_progress(3, total_phases, "Running nnU-Net nodule detection")
        ct_sitk = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_sitk)
        lung_mask_sitk = sitk.ReadImage(lung_mask_path)
        lung_mask_array = sitk.GetArrayFromImage(lung_mask_sitk) > 0

        nodule_mask_raw = create_patches_and_infer(
            ct_array=ct_array,
            lung_mask_array=lung_mask_array,
            original_spacing=ct_sitk.GetSpacing(),
            original_direction=ct_sitk.GetDirection(),
            model_dir=MODEL_DIR,
            temp_dir=temp_dir,
        )

        raw_mask_img = sitk.GetImageFromArray(nodule_mask_raw)
        raw_mask_img.CopyInformation(ct_sitk)
        sitk.WriteImage(raw_mask_img, nodule_mask_raw_path)
        print(f"  Raw nodule mask saved -> {nodule_mask_raw_path}")

        # Phase 6: False positive filtering
        set_progress(4, total_phases, "Filtering false positives (CNN + size)")
        spacing = ct_sitk.GetSpacing()
        voxel_spacing = (spacing[2], spacing[1], spacing[0])

        filtered_mask = filter_false_positives(
            volume=ct_array.astype(np.float32),
            mask=nodule_mask_raw,
            classifier_path=CLASSIFIER_PATH,
            voxel_spacing_mm=voxel_spacing,
        )

        filtered_img = sitk.GetImageFromArray(filtered_mask)
        filtered_img.CopyInformation(ct_sitk)
        sitk.WriteImage(filtered_img, nodule_mask_path)
        print(f"  Filtered nodule mask saved -> {nodule_mask_path}")

        del ct_array, lung_mask_array, nodule_mask_raw, filtered_mask
        del ct_sitk, lung_mask_sitk, raw_mask_img, filtered_img

        # Phase 7a: Feature extraction
        set_progress(5, total_phases, "Extracting nodule features")
        if np.sum(sitk.GetArrayFromImage(sitk.ReadImage(nodule_mask_path)) > 0) == 0:
            update_db("complete", 0)
            set_progress(total_phases, total_phases, "No nodules found after filtering")
            _write_empty_summary(summary_path)
            return

        nodule_rows = extract_all_features(
            volume_path=ct_path,
            mask_path=nodule_mask_path,
            output_dir=nodules_dir,
        )
        num_nodules = len(nodule_rows)

        # Phase 7b: Report generation
        set_progress(6, total_phases, "Generating PDF report")
        report_info = {
            "patient_name": patient_info.get("patient_name", ""),
            "n_dossier": patient_info.get("n_dossier", ""),
            "scan_date": patient_info.get("scan_date", ""),
            "doctor_name": patient_info.get("doctor_name", "Dr. Radiologist"),
            "hospital_name": patient_info.get("hospital_name", "Hopital Habib Bourguiba Sfax"),
            "service": patient_info.get("service", "Service d'Imagerie Medicale"),
        }
        generate_report(summary_path, report_path, report_info)

        # Done
        set_progress(total_phases, total_phases, "Pipeline complete")
        update_db("complete", num_nodules)

    except Exception as e:
        traceback.print_exc()
        update_db("failed", error=str(e))
        pipeline_progress[case_id] = {
            "phase": -1,
            "total_phases": total_phases,
            "message": f"Error: {str(e)}",
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_empty_summary(summary_path):
    import json
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump([], f)
