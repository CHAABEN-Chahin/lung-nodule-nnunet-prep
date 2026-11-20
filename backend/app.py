"""
Nodulis Backend — Complete Flask API
=====================================
Endpoints:
  Auth:    POST /api/auth/login, GET /api/auth/me
  Cases:   GET/POST /api/cases, GET/DELETE /api/cases/<id>
  Pipeline: POST /api/cases/<id>/analyze, GET /api/cases/<id>/status
  Data:    GET /api/cases/<id>/nodules, /volumes/ct, /volumes/lung-mask,
           /volumes/lobe-masks, /volumes/nodule-mask,
           /nodules/<nid>/patch, /nodules/<nid>/mask
  Report:  GET /api/cases/<id>/report
  Stats:   GET /api/stats
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import secrets
import threading
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, send_file, request, g
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DB_PATH, SECRET_KEY, ensure_paths
from database import get_db, init_db
from pipeline.orchestrator import run_pipeline, get_progress

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
CORS(app)


# ── Auth helpers ─────────────────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        db = get_db()
        row = db.execute(
            "SELECT u.* FROM auth_tokens t JOIN users u ON t.user_id = u.id WHERE t.token = ?",
            (token,),
        ).fetchone()
        db.close()
        if not row:
            return jsonify({"error": "Invalid token"}), 401
        g.user = dict(row)
        return f(*args, **kwargs)
    return decorated


# ── Auth endpoints ───────────────────────────────────────────────────

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "")
    password = data.get("password", "")

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        db.close()
        return jsonify({"error": "Invalid credentials"}), 401

    token = secrets.token_hex(32)
    db.execute(
        "INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user["id"], datetime.now().isoformat()),
    )
    db.commit()
    db.close()

    return jsonify({
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "department": user["department"],
            "email": user["email"],
            "hospital_name": user["hospital_name"],
            "service": user["service"],
        },
    })


@app.route("/api/auth/me")
@require_auth
def me():
    u = g.user
    return jsonify({
        "id": u["id"],
        "username": u["username"],
        "full_name": u["full_name"],
        "department": u["department"],
        "email": u["email"],
        "hospital_name": u["hospital_name"],
        "service": u["service"],
    })


@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    db = get_db()
    db.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
    db.commit()
    db.close()
    return jsonify({"ok": True})


@app.route("/api/auth/update-profile", methods=["PUT"])
@require_auth
def update_profile():
    data = request.get_json() or {}
    db = get_db()
    db.execute(
        "UPDATE users SET full_name=?, department=?, email=?, hospital_name=?, service=? WHERE id=?",
        (
            data.get("full_name", g.user["full_name"]),
            data.get("department", g.user["department"]),
            data.get("email", g.user["email"]),
            data.get("hospital_name", g.user["hospital_name"]),
            data.get("service", g.user["service"]),
            g.user["id"],
        ),
    )
    if data.get("new_password"):
        db.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(data["new_password"]), g.user["id"]),
        )
    db.commit()
    db.close()
    return jsonify({"ok": True})


# ── Folder browser ─────────────────────────────────────────────────────

@app.route("/api/browse-folder", methods=["POST"])
@require_auth
def browse_folder():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select DICOM Folder")
    root.destroy()
    if not folder:
        return jsonify({"path": ""})
    return jsonify({"path": folder})


# ── Cases endpoints ──────────────────────────────────────────────────

@app.route("/api/cases", methods=["GET"])
@require_auth
def list_cases():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM cases ORDER BY created_at DESC"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/cases", methods=["POST"])
@require_auth
def create_case():
    data = request.get_json() or {}
    patient_name = data.get("patient_name", "").strip()
    n_dossier = data.get("n_dossier", "").strip()
    scan_date = data.get("scan_date", datetime.now().strftime("%d/%m/%Y"))
    dicom_path = data.get("dicom_path", "").strip()

    if not dicom_path:
        return jsonify({"error": "dicom_path is required"}), 400
    if not os.path.isdir(dicom_path):
        return jsonify({"error": f"DICOM directory not found: {dicom_path}"}), 400

    db = get_db()
    count = db.execute("SELECT COUNT(*) as c FROM cases").fetchone()["c"]
    case_id = f"CS-{datetime.now().year}-{count + 1:05d}"

    db.execute(
        "INSERT INTO cases (case_id, patient_name, n_dossier, scan_date, created_at, status, dicom_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (case_id, patient_name, n_dossier, scan_date, datetime.now().isoformat(), "uploaded", dicom_path),
    )
    db.commit()
    case = dict(db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
    db.close()

    return jsonify(case), 201


@app.route("/api/cases/<case_id>")
@require_auth
def get_case(case_id):
    db = get_db()
    row = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(dict(row))


@app.route("/api/cases/<case_id>", methods=["DELETE"])
@require_auth
def delete_case(case_id):
    db = get_db()
    db.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
    db.commit()
    db.close()
    import shutil
    case_dir = os.path.join(DATA_DIR, case_id)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir, ignore_errors=True)
    return jsonify({"ok": True})


# ── Pipeline endpoints ───────────────────────────────────────────────

@app.route("/api/cases/<case_id>/analyze", methods=["POST"])
@require_auth
def analyze_case(case_id):
    db = get_db()
    row = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "Case not found"}), 404

    case = dict(row)
    if case["status"] == "processing":
        return jsonify({"error": "Analysis already in progress"}), 409

    dicom_path = case["dicom_path"]
    if not os.path.isdir(dicom_path):
        return jsonify({"error": f"DICOM directory not found: {dicom_path}"}), 400

    case_dir = os.path.join(DATA_DIR, case_id)
    u = g.user

    patient_info = {
        "patient_name": case["patient_name"],
        "n_dossier": case["n_dossier"],
        "scan_date": case["scan_date"],
        "doctor_name": u["full_name"],
        "hospital_name": u["hospital_name"],
        "service": u["service"],
    }

    thread = threading.Thread(
        target=run_pipeline,
        args=(case_id, dicom_path, case_dir, DB_PATH, patient_info),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "processing", "case_id": case_id})


@app.route("/api/cases/<case_id>/status")
@require_auth
def case_status(case_id):
    db = get_db()
    row = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "Case not found"}), 404

    case = dict(row)
    progress = get_progress(case_id)

    return jsonify({
        "case_id": case_id,
        "status": case["status"],
        "num_nodules": case["num_nodules"],
        "error_message": case["error_message"],
        "progress": progress,
    })


# ── Data serving endpoints ───────────────────────────────────────────

@app.route("/api/cases/<case_id>/nodules")
@require_auth
def case_nodules(case_id):
    summary_path = os.path.join(DATA_DIR, case_id, "nodules", "summary.json")
    if not os.path.exists(summary_path):
        return jsonify([])
    with open(summary_path) as f:
        return jsonify(json.load(f))


@app.route("/api/cases/<case_id>/nodules/<nodule_id>")
@require_auth
def case_nodule_detail(case_id, nodule_id):
    summary_path = os.path.join(DATA_DIR, case_id, "nodules", "summary.json")
    if not os.path.exists(summary_path):
        return jsonify({"error": "not found"}), 404
    with open(summary_path) as f:
        data = json.load(f)
    for n in data:
        if n.get("nodule_id") == nodule_id:
            return jsonify(n)
    return jsonify({"error": "Nodule not found"}), 404


def _serve_nifti(path, name):
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="application/gzip", as_attachment=False, download_name=name)


@app.route("/api/cases/<case_id>/volumes/ct")
@require_auth
def serve_ct(case_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "ct_volume.nii.gz"),
        f"{case_id}_ct.nii.gz",
    )


@app.route("/api/cases/<case_id>/volumes/lung-mask")
@require_auth
def serve_lung_mask(case_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "lung_mask.nii.gz"),
        f"{case_id}_lung_mask.nii.gz",
    )


@app.route("/api/cases/<case_id>/volumes/lobe-masks")
@require_auth
def serve_lobe_masks(case_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "lobe_masks.nii.gz"),
        f"{case_id}_lobe_masks.nii.gz",
    )


@app.route("/api/cases/<case_id>/volumes/nodule-mask")
@require_auth
def serve_nodule_mask(case_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "nodule_mask.nii.gz"),
        f"{case_id}_nodule_mask.nii.gz",
    )


@app.route("/api/cases/<case_id>/nodules/<nodule_id>/patch")
@require_auth
def serve_nodule_patch(case_id, nodule_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "nodules", nodule_id, "patch_3d.nii.gz"),
        f"{nodule_id}_patch.nii.gz",
    )


@app.route("/api/cases/<case_id>/nodules/<nodule_id>/mask")
@require_auth
def serve_nodule_mask_file(case_id, nodule_id):
    return _serve_nifti(
        os.path.join(DATA_DIR, case_id, "nodules", nodule_id, "mask_3d.nii.gz"),
        f"{nodule_id}_mask.nii.gz",
    )


def _smooth_mask(binary_mask):
    """Dilate + Gaussian smooth a binary mask for clean 3D contours."""
    import scipy.ndimage
    import numpy as np
    dilated = scipy.ndimage.binary_dilation(binary_mask, iterations=5)
    smoothed = scipy.ndimage.gaussian_filter(dilated.astype(np.float32), sigma=3.0)
    return smoothed > 0.3


@app.route("/api/cases/<case_id>/volumes/ct-lungs")
@require_auth
def serve_ct_lungs(case_id):
    case_dir = os.path.join(DATA_DIR, case_id)
    cached = os.path.join(case_dir, "ct_lungs_smooth.nii.gz")
    if not os.path.exists(cached):
        ct_path = os.path.join(case_dir, "ct_volume.nii.gz")
        mask_path = os.path.join(case_dir, "lung_mask.nii.gz")
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            return jsonify({"error": "Files not found"}), 404
        import SimpleITK as sitk
        import numpy as np
        ct = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.int16)
        mask_arr = sitk.GetArrayFromImage(mask)
        smooth = _smooth_mask(mask_arr > 0)
        ct_arr[~smooth] = -2000
        out = sitk.GetImageFromArray(ct_arr)
        out.CopyInformation(ct)
        sitk.WriteImage(out, cached)
    return _serve_nifti(cached, f"{case_id}_ct_lungs.nii.gz")


@app.route("/api/cases/<case_id>/volumes/ct-lobe/<int:label>")
@require_auth
def serve_ct_lobe(case_id, label):
    case_dir = os.path.join(DATA_DIR, case_id)
    cached = os.path.join(case_dir, f"ct_lobe_{label}_smooth.nii.gz")
    if not os.path.exists(cached):
        ct_path = os.path.join(case_dir, "ct_volume.nii.gz")
        lobe_path = os.path.join(case_dir, "lobe_masks.nii.gz")
        if not os.path.exists(ct_path) or not os.path.exists(lobe_path):
            return jsonify({"error": "Files not found"}), 404
        import SimpleITK as sitk
        import numpy as np
        ct = sitk.ReadImage(ct_path)
        lobes = sitk.ReadImage(lobe_path)
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.int16)
        lobe_arr = sitk.GetArrayFromImage(lobes)
        smooth = _smooth_mask(lobe_arr == label)
        ct_arr[~smooth] = -2000
        out = sitk.GetImageFromArray(ct_arr)
        out.CopyInformation(ct)
        sitk.WriteImage(out, cached)
    return _serve_nifti(cached, f"{case_id}_lobe_{label}.nii.gz")


@app.route("/api/cases/<case_id>/lobe-info")
@require_auth
def lobe_info(case_id):
    case_dir = os.path.join(DATA_DIR, case_id)
    lobe_path = os.path.join(case_dir, "lobe_masks.nii.gz")
    if not os.path.exists(lobe_path):
        return jsonify({"error": "Lobe masks not found"}), 404
    import SimpleITK as sitk
    import numpy as np
    lobes = sitk.ReadImage(lobe_path)
    lobe_arr = sitk.GetArrayFromImage(lobes)
    labels = sorted([int(x) for x in np.unique(lobe_arr) if x > 0])
    lobe_meta = {
        1: ("Upper Left", "left"), 2: ("Lower Left", "left"),
        3: ("Upper Right", "right"), 4: ("Middle Right", "right"), 5: ("Lower Right", "right"),
        10: ("Upper Left", "left"), 11: ("Lower Left", "left"),
        12: ("Upper Right", "right"), 13: ("Middle Right", "right"), 14: ("Lower Right", "right"),
    }
    result = []
    for lab in labels:
        name, side = lobe_meta.get(lab, (f"Lobe {lab}", "unknown"))
        result.append({
            "label": lab,
            "name": name,
            "side": side,
            "voxels": int(np.sum(lobe_arr == lab)),
        })
    return jsonify(result)


# ── Report endpoint ──────────────────────────────────────────────────

@app.route("/api/cases/<case_id>/report")
@require_auth
def case_report(case_id):
    case_dir = os.path.join(DATA_DIR, case_id)
    report_path = os.path.join(case_dir, "report.pdf")

    if not os.path.exists(report_path):
        summary_path = os.path.join(case_dir, "nodules", "summary.json")
        if not os.path.exists(summary_path):
            return jsonify({"error": "No analysis data available"}), 404

        db = get_db()
        row = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        db.close()
        if not row:
            return jsonify({"error": "Case not found"}), 404
        case = dict(row)
        u = g.user

        from pipeline.generate_report import generate_report
        generate_report(summary_path, report_path, {
            "patient_name": case["patient_name"],
            "n_dossier": case["n_dossier"],
            "scan_date": case["scan_date"],
            "doctor_name": u["full_name"],
            "hospital_name": u["hospital_name"],
            "service": u["service"],
        })

    return send_file(
        report_path, mimetype="application/pdf",
        as_attachment=True, download_name=f"rapport_{case_id}.pdf",
    )


# ── Stats endpoint ───────────────────────────────────────────────────

@app.route("/api/stats")
@require_auth
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM cases").fetchone()["c"]
    complete = db.execute("SELECT COUNT(*) as c FROM cases WHERE status='complete'").fetchone()["c"]
    processing = db.execute("SELECT COUNT(*) as c FROM cases WHERE status='processing'").fetchone()["c"]
    total_nodules = db.execute("SELECT COALESCE(SUM(num_nodules), 0) as s FROM cases").fetchone()["s"]
    db.close()
    return jsonify({
        "total_cases": total,
        "complete_cases": complete,
        "processing_cases": processing,
        "total_nodules": total_nodules,
    })


# ── Health check ─────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "data_dir": DATA_DIR})


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    ensure_paths()
    init_db()
    print(f"Nodulis Backend starting...")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Database: {DB_PATH}")
    print(f"  All paths are ASCII-safe (no Unicode issues)")
    app.run(debug=False, port=5000, threaded=True)
