# TL-MaskNet – Real-Time Face Mask Detection
A deep-learning based face mask detection system built using MobileNetV2, TensorFlow, and OpenCV.
The model detects whether a person is wearing a mask or not in real time using webcam video.

🚀 Features
✔ Real-time webcam mask detection
✔ Uses MobileNetV2 (lightweight + fast)
✔ Classifies: with_mask vs without_mask
✔ Bounding boxes (Green = Mask, Red = No Mask)
✔ Custom dataset support
✔ Easy-to-run scripts (train.py, detect.py)

📂 Project Structure
TL-MaskNet/
│── dataset/
│   ├── train/
│   │   ├── with_mask/
│   │   └── without_mask/
│   └── val/
│       ├── with_mask/
│       └── without_mask/
│
│── model/
│   └── face_mask_mobilenetv2.h5
│
│── train.py
│── detect.py
│── README.md
│── requirements.txt

📦 Requirements
Install dependencies:
pip install tensorflow opencv-python numpy matplotlib
Or install from requirements file:
pip install -r requirements.txt

🧠 Training the Model
Place your dataset inside:
dataset/train/
dataset/val/

Then run:
python train.py
After training, the model will be saved automatically in:
model/face_mask_mobilenetv2.h5

🎥 Running Real-Time Mask Detection
Start webcam detection:
python detect.py

Output:
😷 Green box → with_mask
🙂 Red box → without_mask
Press Q to quit the window.

🧬 Model
Base CNN: MobileNetV2 (ImageNet weights)
Fine-tuned using custom dataset
Loss: Binary Crossentropy
Optimizer: Adam

📝 License
This project is for educational and assignment purposes.

👤 Author
Prabhakara Rao M
GitHub: https://github.com/prabhameesala3
