# app.py
"""
Automated Attendance System for Rural Schools
Single-file Streamlit app (Python only)

Features:
- Register students (capture photo via webcam or upload one).
- Train a lightweight LBPH face recognizer from registered images.
- Mark attendance by recognizing faces via webcam.
- RFID-mode simulation (enter ID) as fallback.
- Saves attendance CSVs by date and keeps a small audit trail.

Notes:
- Uses OpenCV's Haar cascade to detect faces and LBPHFaceRecognizer for recognition.
- Designed to be simple to deploy on low-resource machines.
"""

import streamlit as st
import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import tempfile
import shutil

# -----------------------
# Configuration / paths
# -----------------------
BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "dataset"         # each student: dataset/{student_id}_{name}/imgX.jpg
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
TRAINER_PATH = MODEL_DIR / "trainer.yml"
ATTENDANCE_DIR = BASE_DIR / "attendance"   # attendance/YYYY-MM-DD.csv
ATTENDANCE_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    ATTENDANCE_DIR.mkdir(exist_ok=True)

def save_student_image(student_id: str, student_name: str, image_bytes: bytes):
    """
    Save uploaded image bytes into dataset directory path:
    dataset/{student_id}_{student_name}/img_{n}.jpg
    """
    folder = DATASET_DIR / f"{student_id}_{student_name.replace(' ', '_')}"
    folder.mkdir(parents=True, exist_ok=True)
    # generate next index
    existing = list(folder.glob("img_*.jpg"))
    n = len(existing) + 1
    fname = folder / f"img_{n}.jpg"
    with open(fname, "wb") as f:
        f.write(image_bytes)
    return str(fname)

def detect_face_from_imagefile(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    # choose the largest face
    x,y,w,h = sorted(faces, key=lambda rect: rect[2]*rect[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))
    return face

def gather_training_data():
    """
    Scan dataset folders, detect faces, return (faces, labels, label_map)
    label_map: label_int -> (student_id, name)
    """
    faces = []
    labels = []
    label_map = {}
    label_counter = 0
    for student_folder in sorted(DATASET_DIR.iterdir()):
        if not student_folder.is_dir():
            continue
        # folder name expected: {id}_{name}
        folder_name = student_folder.name
        try:
            student_id, student_name = folder_name.split("_", 1)
        except ValueError:
            student_id = folder_name
            student_name = ""
        label_counter += 1
        label = label_counter
        label_map[label] = (student_id, student_name.replace("_", " "))
        # iterate images
        for img_path in student_folder.glob("*.jpg"):
            try:
                face = detect_face_from_imagefile(img_path)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
            except Exception as e:
                print("Error processing", img_path, e)
    return faces, labels, label_map

def train_and_save_model():
    faces, labels, label_map = gather_training_data()
    if len(faces) == 0:
        raise ValueError("No faces found in dataset. Register at least one student with a clear face image.")
    # create LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(str(TRAINER_PATH))
    # save label map as csv
    label_map_df = pd.DataFrame([
        {"label": lab, "student_id": sid, "name": name}
        for lab,(sid,name) in label_map.items()
    ])
    label_map_df.to_csv(MODEL_DIR / "labels.csv", index=False)
    return len(faces), len(label_map)

def load_trained_model():
    if not TRAINER_PATH.exists():
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(TRAINER_PATH))
    # load label map
    label_df = pd.read_csv(MODEL_DIR / "labels.csv")
    label_map = {int(row["label"]):(row["student_id"], row["name"]) for _,row in label_df.iterrows()}
    return recognizer, label_map

def mark_attendance(student_id, name, method="face"):
    today = date.today().isoformat()
    csv_path = ATTENDANCE_DIR / f"{today}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["timestamp","student_id","name","method"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # avoid duplicate mark within same day for same student
    if not ((df["student_id"] == student_id) & (df["name"] == name)).any():
        df = pd.concat([df, pd.DataFrame([{"timestamp":now,"student_id":student_id,"name":name,"method":method}])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        return True
    return False

# -----------------------
# Streamlit interface
# -----------------------
st.set_page_config(page_title="Automated Attendance (Rural Schools)", layout="wide")
ensure_dirs()

st.title("Automated Attendance System — Low-cost (Streamlit + Python)")

tabs = st.tabs(["Register Student", "Train Model", "Mark Attendance (Camera)", "RFID Fallback", "Admin / Attendance"])

# ---------- Register Student ----------
with tabs[0]:
    st.header("Register Student (capture or upload photo)")
    with st.form("reg_form"):
        col1, col2 = st.columns([1,2])
        with col1:
            student_id = st.text_input("Student ID (unique)", value="")
            student_name = st.text_input("Student Name", value="")
            capture_mode = st.radio("How to provide photo?", ("Use Webcam", "Upload Image"), index=0)
        with col2:
            st.write("Tips: Face should be clear, front-facing. Capture multiple photos for better accuracy.")
            if capture_mode == "Use Webcam":
                img_file = st.camera_input("Take a photo")
            else:
                img_file = st.file_uploader("Upload a photo (jpg/png)", type=["jpg","jpeg","png"])
        submitted = st.form_submit_button("Register")
    if submitted:
        if not student_id.strip():
            st.error("Student ID is required.")
        elif img_file is None:
            st.error("Please provide a photo (camera or upload).")
        else:
            # read bytes and detect face
            bytes_img = img_file.getvalue()
            saved = save_student_image(student_id.strip(), student_name.strip() or "unknown", bytes_img)
            st.success(f"Saved image to {saved}")
            # show detection
            face = detect_face_from_imagefile(saved)
            if face is None:
                st.warning("No face detected in the saved image. Consider re-taking/uploading a clearer front-facing photo.")
            else:
                st.image(cv2.resize(face, (200,200)), caption="Detected face (grayscale preview)", use_column_width=False)
                st.info("Image saved to dataset. Run 'Train Model' after registering students.")

# ---------- Train Model ----------
with tabs[1]:
    st.header("Train LBPH Face Recognizer")
    st.write("Train from images in `dataset/`. Each student folder should be `id_name` containing `img_*.jpg`.")
    st.write("Training creates a lightweight model at `model/trainer.yml` and `model/labels.csv`.")
    if st.button("Start Training"):
        try:
            n_faces, n_students = train_and_save_model()
            st.success(f"Training complete — {n_faces} face samples across {n_students} students. Model saved to {TRAINER_PATH}")
        except Exception as e:
            st.error(f"Training failed: {e}")
    st.write("Model status:")
    if TRAINER_PATH.exists():
        st.success("Model exists: " + str(TRAINER_PATH))
        st.write("Labels:")
        if (MODEL_DIR / "labels.csv").exists():
            st.dataframe(pd.read_csv(MODEL_DIR / "labels.csv"))
    else:
        st.warning("No trained model found. Please register students and run training.")

# ---------- Mark Attendance (Camera) ----------
with tabs[2]:
    st.header("Mark Attendance using Camera (face recognition)")
    st.info("Point the webcam at the student. The model will attempt to detect and recognize the face.")
    recognizer, label_map = load_trained_model()
    if recognizer is None:
        st.warning("No trained model found. Please train the model first.")
    else:
        # We'll use camera input capture each attempt
        img_input = st.camera_input("Capture to mark attendance")
        if img_input is not None:
            # save to temp file and detect
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(img_input.getvalue()); tmp.flush()
            face = detect_face_from_imagefile(tmp.name)
            if face is None:
                st.error("No face detected. Ensure face is clearly visible and well-lit.")
            else:
                # predict
                label, confidence = recognizer.predict(face)
                sid, sname = label_map.get(label, ("unknown","unknown"))
                st.write(f"Prediction: {sid} — {sname}")
                st.write(f"Confidence (lower is better): {confidence:.2f}")
                # set a confidence threshold (tuneable)
                threshold = st.slider("Recognition threshold (lower is stricter)", min_value=30, max_value=150, value=70)
                if confidence <= threshold:
                    marked = mark_attendance(sid, sname, method="face")
                    if marked:
                        st.success(f"Attendance marked for {sname} ({sid}) at {datetime.now().strftime('%H:%M:%S')}")
                    else:
                        st.info(f"{sname} ({sid}) already marked today.")
                else:
                    st.warning("Face not recognized with enough confidence. Try again or use RFID fallback.")

# ---------- RFID Fallback ----------
with tabs[3]:
    st.header("RFID Fallback (Simulated)")
    st.write("If a hardware RFID reader is not available, teachers can enter the Student ID manually here.")
    with st.form("rfid_form"):
        rfid_id = st.text_input("Student ID (from RFID card or manual entry)")
        rfid_name = st.text_input("Student name (optional)")
        method = st.selectbox("Method", ["rfid", "manual"])
        btn = st.form_submit_button("Mark Attendance")
    if btn:
        if not rfid_id.strip():
            st.error("Student ID required.")
        else:
            # try to find name from dataset or labels
            found_name = None
            # attempt to find folder with id
            for folder in DATASET_DIR.iterdir():
                if folder.is_dir() and folder.name.startswith(rfid_id + "_"):
                    # extract name from folder name
                    try:
                        _, nm = folder.name.split("_",1)
                        found_name = nm.replace("_"," ")
                    except:
                        found_name = rfid_name or "unknown"
                    break
            name_to_use = rfid_name.strip() or found_name or "unknown"
            marked = mark_attendance(rfid_id.strip(), name_to_use, method=method)
            if marked:
                st.success(f"Marked attendance for {name_to_use} ({rfid_id})")
            else:
                st.info("Attendance already marked for this student today.")

# ---------- Admin / Attendance ----------
with tabs[4]:
    st.header("Admin / View Attendance")
    st.write("Download or view attendance CSVs stored in `attendance/` folder.")
    files = sorted(ATTENDANCE_DIR.glob("*.csv"), reverse=True)
    selected = st.selectbox("Select date file", [f.name for f in files] if files else ["No files"])
    if files:
        sel_path = ATTENDANCE_DIR / selected
        df = pd.read_csv(sel_path)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=selected, mime="text/csv")
    # quick summary
    if st.button("Show Today Summary"):
        today_path = ATTENDANCE_DIR / f"{date.today().isoformat()}.csv"
        if today_path.exists():
            st.table(pd.read_csv(today_path))
        else:
            st.info("No attendance marked today yet.")

# -----------------------
# Helpful tips
# -----------------------
st.sidebar.header("Deployment & Tips")
st.sidebar.markdown("""
- Install dependencies (see requirements.txt).  
- Prefer consistent lighting and front-facing photos for better accuracy.  
- Encourage registering 3–5 photos per student (different angles/lighting) to improve recognition.  
- For very low-resource sites, use RFID/manual mode which needs no camera.  
- To deploy on a local machine: `streamlit run app.py` and open the browser on the device.
""")
st.sidebar.header("File layout (created automatically)")
st.sidebar.code(f"""
{BASE_DIR}/dataset/              # student folders: id_name/img_*.jpg
{BASE_DIR}/model/trainer.yml
{BASE_DIR}/model/labels.csv
{BASE_DIR}/attendance/YYYY-MM-DD.csv
""")
