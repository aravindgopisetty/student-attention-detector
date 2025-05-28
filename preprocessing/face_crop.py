# preprocessing/face_crop.py\import os
import cv2
from deepface.detectors import FaceDetector

def crop_faces(input_dir, output_dir, label):
    detector = FaceDetector.build_model("opencv")
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    for img_name in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, img_name))
        detections = FaceDetector.detect_faces(detector, img, align=True)
        for i,(face,_) in enumerate(detections):
            face_resized = cv2.resize(face, (224,224))
            cv2.imwrite(
              os.path.join(output_dir, label, f"{os.path.splitext(img_name)[0]}_{i}.jpg"),
              face_resized
            )

if __name__ == '__main__':
    raw = "../data/raw_frames"
    out = "../data_crops"
    crop_faces(raw + "/attentive", out, "paying_attention")
    crop_faces(raw + "/distracted", out, "distracted")