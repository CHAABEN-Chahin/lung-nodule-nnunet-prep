import sqlite3
import os
from werkzeug.security import generate_password_hash
from config import DB_PATH


def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL DEFAULT '',
            department TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            hospital_name TEXT NOT NULL DEFAULT '',
            service TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT UNIQUE NOT NULL,
            patient_name TEXT NOT NULL DEFAULT '',
            n_dossier TEXT NOT NULL DEFAULT '',
            scan_date TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            dicom_path TEXT NOT NULL DEFAULT '',
            num_nodules INTEGER NOT NULL DEFAULT 0,
            error_message TEXT DEFAULT NULL
        );
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", ("admin",)
    ).fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users (username, password_hash, full_name, department, email, hospital_name, service) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "admin",
                generate_password_hash("admin"),
                "Dr. Radiologist",
                "Radiology Dept.",
                "radiologist@hospital.local",
                "Hopital Habib Bourguiba Sfax",
                "Service d'Imagerie Medicale",
            ),
        )
    db.commit()
    db.close()
