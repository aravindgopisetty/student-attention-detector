import os, time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
INPUT_SIZE = 224
ATTENTION_THRESHOLD = 0.5     # model sigmoid cutoff
SMOOTH_FRAMES = 10            # smoothing window
MAX_FACES = 10                # max faces per frame

# pose-adjust parameters
YAW_DEADZONE = 10.0           # degrees within which no adjustment
YAW_MAX = 45.0                # degrees at which attention reduces to 0

# ─── PATHS ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
SCORE_FILE = os.path.join(BASE_DIR, 'flask_app', 'attention_score.txt')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'trained_models', 'attention_model_best_epoch.h5')

# ─── MODEL LOADING ───────────────────────────────────────────────────────────────
def load_attention_model():
    inp = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inp)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ─── FACE DETECTION (Haar Cascade) ───────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── MEDIAPIPE MESH FOR POSE ──────────────────────────────────────────────────────
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=MAX_FACES,
                                           refine_landmarks=True, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
# 3D model points for solvePnP (nose, chin, eye corners, mouth corners)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype='double')

# ─── HEAD POSE UTILITY ───────────────────────────────────────────────────────────
def estimate_yaw(landmarks, w, h):
    try:
        pts2d = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype='double')
        cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype='double')
        dist = np.zeros((4,1))
        ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, pts2d, cam, dist)
        if not ok:
            return 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return angles[1]  # yaw
    except:
        return 0.0

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
def main():
    model = load_attention_model()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    history = deque(maxlen=SMOOTH_FRAMES)
    cv2.namedWindow('Classroom Attention', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

        total, attentive = 0, 0
        mesh_res = mp_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks_list = mesh_res.multi_face_landmarks or []

        for i, (x, y, fw, fh) in enumerate(faces[:MAX_FACES]):
            # get corresponding mesh landmarks if available
            landmarks = landmarks_list[i].landmark if i < len(landmarks_list) else None
            face_img = frame[y:y+fh, x:x+fw]
            if face_img.size == 0:
                continue
            total += 1
            inp = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE)) / 255.0
            score = model.predict(np.expand_dims(inp,0), verbose=0)[0][0]
            yaw = estimate_yaw(landmarks, w, h) if landmarks else 0.0
            # adjust score linearly based on yaw
            yaw_deg = abs(yaw)
            if yaw_deg > YAW_DEADZONE:
                reduction = min((yaw_deg - YAW_DEADZONE) / (YAW_MAX - YAW_DEADZONE), 1.0)
                score *= (1 - reduction)
            is_att = score > ATTENTION_THRESHOLD
            if is_att:
                attentive += 1
            col = (0,255,0) if is_att else (0,0,255)
            cv2.rectangle(frame,(x,y),(x+fw,y+fh),col,2)
            cv2.putText(frame,f"{score:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)

        pct = (attentive/total*100) if total>0 else 0.0
        history.append(pct)
        avg_pct = np.mean(history) if history else 0.0
        cv2.putText(frame,f"Class Attention: {avg_pct:.1f}% ({attentive}/{total})",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,0),2)
        try:
            with open(SCORE_FILE,'w') as f:
                f.write(f"{avg_pct:.1f}")
        except Exception as e:
            print(f"Score write error: {e}")
        cv2.imshow('Classroom Attention', frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()






