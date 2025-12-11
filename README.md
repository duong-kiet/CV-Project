# 🍽️ Hệ thống Hỗ trợ Phục vụ Khách hàng - AI Emotion Detection

Hệ thống AI thông minh sử dụng computer vision và machine learning để nhận diện cảm xúc khách hàng real-time, từ đó đưa ra hướng dẫn cụ thể cho nhân viên phục vụ nhà hàng.

## ✨ Tính năng chính

- **🎯 Real-time Face Detection**: Phát hiện khuôn mặt khách hàng từ camera sử dụng YOLO v12
- **👤 Face Recognition**: Nhận diện khách hàng quen bằng FaceNet embedding (512-D vector)
- **😊 Emotion Detection**: Phân tích 8 loại cảm xúc (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral, Contempt) sử dụng MLT model
- **🤖 AI Assistant**: Gemini AI phân tích cảm xúc và đưa ra hướng dẫn phục vụ cụ thể cho nhân viên
- **📊 Emotion History**: Lưu lịch sử cảm xúc theo từng khách hàng (face_id) vào SQLite
- **🔊 Text-to-Speech**: Tự động đọc hướng dẫn bằng tiếng Việt
- **🔄 Auto Detection**: Tự động detect cảm xúc theo khoảng thời gian định kỳ

## 🏗️ Kiến trúc hệ thống

```
Camera Stream
    ↓
YOLO Face Detection → FaceNet Embedding → Face ID Recognition
    ↓
MLT Emotion Model → Emotion Prediction
    ↓
SQLite Database → Emotion History + Face Embeddings
    ↓
Gemini AI → Service Recommendations
    ↓
TTS → Audio Output
```

## 📋 Yêu cầu hệ thống

- Python 3.10+
- Webcam/Camera
- GPU (khuyến nghị, không bắt buộc)
- Gemini API Key (từ Google AI Studio)

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd DeepFace
```

### 2. Tạo virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi với NumPy/TensorFlow, cài đặt:

```bash
pip install "numpy<2" "protobuf>=3.20.2,<6.0"
```

### 4. Tải model files

Đảm bảo có các file sau:
- `yolov12l-face.onnx` - YOLO face detection model (đã có trong repo)
- `services/stage2_epoch_7_loss_1.1606_acc_0.5589.pth` - MLT emotion model (đã có trong repo)
- FaceNet model sẽ tự động download khi chạy lần đầu

## ⚙️ Cấu hình

### 1. Tạo file `.env`

Tạo file `.env` trong thư mục gốc:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Cấu hình database

Database SQLite sẽ tự động được tạo tại `database/emotion_memory.db` khi chạy lần đầu.

Schema:
- `id`: Primary key
- `user_id`: Face ID của khách hàng
- `timestamp`: Thời điểm detect
- `dominant_emotion`: Cảm xúc chính
- `emotions_json`: Chi tiết xác suất các cảm xúc
- `similarity`: Độ tương đồng với face đã biết
- `face_embedding`: Vector đặc trưng khuôn mặt (BLOB)
- `box_json`: Tọa độ bounding box

## 🎮 Sử dụng

### Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### Hướng dẫn sử dụng

1. **Bật camera**: Click "Start" trong WebRTC component
2. **Chọn chế độ**:
   - **Tự động detect**: Tự động phân tích cảm xúc mỗi 15 giây
   - **Detect ngay**: Nhấn nút để detect ngay lập tức
3. **Xem kết quả**:
   - Face ID và similarity score
   - Cảm xúc chính và chi tiết xác suất
   - Hướng dẫn phục vụ từ AI
4. **Text-to-Speech**: Bật checkbox để nghe audio hướng dẫn

### Các nút điều khiển

- **🔄 Reset và bắt đầu lại**: Xóa lịch sử, reset face database
- **▶️ Detect cảm xúc ngay**: Trigger detection ngay lập tức
- **🔄 Tự động detect**: Bật/tắt chế độ tự động
- **🔊 Bật đọc text-to-speech**: Bật/tắt TTS

## 📁 Cấu trúc project

```
DeepFace/
├── app.py                          # Streamlit main app
├── components/
│   └── camera_auto.py              # Camera auto detection component
├── services/
│   ├── deepface_service.py         # Emotion prediction (MLT model)
│   ├── face_recognition_service.py # Face detection & recognition (YOLO + FaceNet)
│   ├── emotion_agent_service.py    # Gemini AI integration
│   ├── vector_db_service.py        # SQLite database operations
│   ├── gemini_service.py           # Gemini API wrapper
│   ├── tts_service.py              # Text-to-speech (Edge TTS)
│   ├── MLT.py                      # MLT emotion model
│   └── stage2_epoch_7_loss_1.1606_acc_0.5589.pth  # Emotion model weights
├── database/
│   └── emotion_memory.db           # SQLite database (auto-generated)
├── test/
│   └── app.py                      # Test script (OpenCV standalone)
├── yolov12l-face.onnx              # YOLO face detection model
└── requirements.txt               # Python dependencies
```

## 🔧 Tech Stack

### Computer Vision
- **YOLO v12**: Face detection
- **FaceNet (InceptionResnetV1)**: Face recognition & embedding
- **MLT Model**: Emotion classification (8 emotions)
- **OpenCV**: Image processing

### AI/ML
- **Google Gemini**: Natural language generation cho service recommendations
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing

### Backend
- **Streamlit**: Web framework
- **Streamlit-WebRTC**: Real-time video streaming
- **SQLite**: Database lưu emotion history & face embeddings

### TTS
- **Edge TTS**: Text-to-speech tiếng Việt

## 🎯 Workflow

1. **Face Detection**: YOLO detect tất cả khuôn mặt trong frame
2. **Face Recognition**: 
   - Tính FaceNet embedding (512-D)
   - So sánh với database (similarity threshold = 0.7)
   - Nếu match → trả về face_id cũ (khách quen)
   - Nếu không match → tạo face_id mới (khách mới)
3. **Emotion Prediction**: MLT model predict 8 emotions
4. **Database Storage**: Lưu emotion + face embedding + metadata
5. **AI Analysis**: Gemini phân tích cảm xúc và đưa hướng dẫn
6. **TTS Output**: Đọc hướng dẫn bằng tiếng Việt
