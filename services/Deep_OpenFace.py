import cv2
import torch
import numpy as np
from deepface import DeepFace
from deepface.modules import detection
from torchvision import transforms
from PIL import Image

from services.MLT import MLT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

expression_labels = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

emotion_model = MLT()
emotion_model.load_state_dict(torch.load(
    "C:/Users/Admin/PycharmProjects/CV-Project/stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
    map_location=device
))
emotion_model.to(device)
emotion_model.eval()


def predict_emotion(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tens = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emotion_output, _, _ = emotion_model(tens)

    idx = torch.argmax(emotion_output).item()
    return expression_labels[idx]


def run_realtime():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using DeepFace
        try:
            faces = detection.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=True
            )
        except:
            faces = []

        # Xử lý từng khuôn mặt
        for f in faces:
            fa = f["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            emotion = predict_emotion(face)

            # Vẽ bounding box + emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Realtime Emotion (PyTorch + DeepFace Detector)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run_realtime()
