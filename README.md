# 🎯 Student Attention Detector

This project is a real-time AI system that detects whether students are paying attention during a class session using a webcam. It uses a fine-tuned MobileNetV2 model and head-pose estimation to calculate attention scores for multiple people in the frame and displays the overall classroom attention percentage on a live dashboard.

---

## 🚀 Features

- 🔍 Real-time multi-face detection (MediaPipe + Haar Cascades)
- 🧠 Attention classification using a custom-trained MobileNetV2
- 📏 Head-pose correction using SolvePnP to handle yaw/pose sensitivity
- 📊 Real-time classroom attention percentage (smoothed)
- 🖥️ Flask dashboard to view updated score every second

---

## 📂 Project Structure

student_attention_detector/
├── scripts/
│ └── inference.py # Real-time detection + model inference
├── flask_app/
│ ├── app.py # Flask server
│ ├── templates/
│ │ └── index.html # Web dashboard template
│ └── attention_score.txt # Shared score (runtime-generated)
├── trained_models/
│ └── attention_model_best_epoch.h5 # Trained model weights
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


---

## ⚙️ Setup Instructions

1. Clone the repo:

```bash
git clone https://github.com/aravindgopisetty/student-attention-detector.git
cd student-attention-detector
2. Create a virtual environment:
python -m venv venv
venv\\Scripts\\activate    # On Windows
3.Install dependencies:
pip install -r requirements.txt

Run the Application
Step 1: Start the real-time detector
bash
Copy
Edit
python scripts/inference.py
This will:

Open your webcam

Detect faces

Estimate attention scores

Write the current attention % to attention_score.txt

Step 2: Launch the dashboard
In another terminal:

bash
Copy
Edit
cd flask_app
python app.py
Open in browser: http://127.0.0.1:5000

You’ll see the current attention score auto-refreshing every second.

📦 Requirements
Python 3.7+

TensorFlow / Keras

OpenCV

MediaPipe

Flask

NumPy

📸 Demo
Coming soon — upload a screen recording or GIF to show the system in action!

📄 License
This project is under the MIT License — feel free to use and adapt it with attribution.

🙌 Acknowledgements
MediaPipe

TensorFlow

OpenCV

Trained on custom student attention dataset

yaml
Copy
Edit












