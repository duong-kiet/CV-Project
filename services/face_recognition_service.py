"""
Face Recognition Service
Sử dụng YOLO để detect faces và FaceNet để nhận diện khuôn mặt.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet.facenet_pytorch import InceptionResnetV1
from typing import List, Dict, Any, Optional, Tuple

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# GLOBAL MODELS (lazy loading)
# -------------------------------
_yolo_detector: Optional[YOLO] = None
_facenet_model: Optional[InceptionResnetV1] = None

# Face database - lưu embedding của các khuôn mặt đã biết
_face_db: Dict[int, np.ndarray] = {}  # {face_id: embedding}
_next_face_id: int = 0
_db_loaded: bool = False  # Flag để biết đã load từ DB chưa


def _get_yolo_detector() -> YOLO:
    """Lazy load YOLO detector."""
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLO("./yolov12l-face.onnx", task="detect")
    return _yolo_detector


def _get_facenet_model() -> InceptionResnetV1:
    """Lazy load FaceNet model."""
    global _facenet_model
    if _facenet_model is None:
        _facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _facenet_model


def _load_face_db_from_database():
    """
    Load face embeddings từ SQLite database vào _face_db.
    Gọi 1 lần khi app khởi động để khôi phục danh sách khách đã biết.
    """
    global _face_db, _next_face_id, _db_loaded
    
    if _db_loaded:
        return  # Đã load rồi, không cần load lại
    
    try:
        from services.vector_db_service import get_all_user_embeddings
        
        saved_embeddings = get_all_user_embeddings()
        
        if saved_embeddings:
            _face_db = saved_embeddings
            # Tìm max face_id để set _next_face_id
            max_id = max(saved_embeddings.keys()) if saved_embeddings else -1
            _next_face_id = max_id + 1
            print(f"[FaceRecognition] Loaded {len(saved_embeddings)} faces from database. Next ID: {_next_face_id}")
        else:
            print("[FaceRecognition] No saved faces in database. Starting fresh.")
        
        _db_loaded = True
        
    except Exception as e:
        print(f"[FaceRecognition] Error loading from database: {e}")
        _db_loaded = True  # Đánh dấu đã thử load để không retry liên tục


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tính cosine similarity giữa 2 embedding vectors."""
    return float(np.dot(a, b))


def get_face_embedding(face_img: np.ndarray) -> np.ndarray:
    """
    Tính FaceNet embedding từ ảnh khuôn mặt (BGR format).
    
    Args:
        face_img: Ảnh khuôn mặt đã crop, format BGR (OpenCV)
    
    Returns:
        Embedding vector đã normalize (512-D)
    """
    facenet = _get_facenet_model()
    
    # Resize về 160x160 (input size của FaceNet)
    face_img = cv2.resize(face_img, (160, 160))
    # BGR -> RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    face_img = face_img / 255.0
    # HWC -> CHW
    face_img = np.transpose(face_img, (2, 0, 1))
    
    # Convert to tensor
    face_tensor = torch.tensor(face_img).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        emb = facenet(face_tensor).cpu().numpy()[0]
    
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb


def identify_face(embedding: np.ndarray, threshold: float = 0.7) -> Tuple[int, float]:
    """
    Nhận diện khuôn mặt từ embedding.
    Nếu không match với ai trong database → đăng ký face mới.
    
    Args:
        embedding: Face embedding vector
        threshold: Ngưỡng similarity để coi là cùng người
    
    Returns:
        Tuple (face_id, similarity_score)
    """
    global _face_db, _next_face_id
    
    # Load từ database nếu chưa load
    _load_face_db_from_database()
    
    best_id = None
    best_sim = -1.0
    
    # Tìm face giống nhất trong database
    for fid, saved_emb in _face_db.items():
        sim = cosine_similarity(embedding, saved_emb)
        if sim > best_sim:
            best_sim = sim
            best_id = fid
    
    # Nếu không đủ giống → đăng ký face mới
    if best_sim < threshold:
        new_id = _next_face_id
        _face_db[new_id] = embedding
        _next_face_id += 1
        print(f"[FaceRecognition] New face registered: ID={new_id}, best_sim={best_sim:.3f}")
        return new_id, best_sim
    
    print(f"[FaceRecognition] Face matched: ID={best_id}, similarity={best_sim:.3f}")
    return best_id, best_sim


def detect_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect tất cả khuôn mặt trong frame bằng YOLO.
    
    Args:
        frame: Frame ảnh BGR từ camera
    
    Returns:
        List các dict chứa thông tin face:
        [{"box": [x1, y1, x2, y2], "confidence": float, "face_img": np.ndarray}, ...]
    """
    detector = _get_yolo_detector()
    results = detector(frame)[0]
    
    faces = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) if box.conf is not None else 1.0
        
        # Crop face
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        
        faces.append({
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "face_img": face_img
        })
    
    return faces


def detect_and_identify_faces(frame: np.ndarray, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Detect và nhận diện tất cả khuôn mặt trong frame.
    
    Args:
        frame: Frame ảnh BGR từ camera
        threshold: Ngưỡng similarity cho face recognition
    
    Returns:
        List các dict chứa thông tin đầy đủ:
        [{
            "face_id": int,
            "similarity": float,
            "box": [x1, y1, x2, y2],
            "confidence": float,
            "face_img": np.ndarray,
            "embedding": np.ndarray,
            "area": int  # diện tích bounding box
        }, ...]
    """
    faces = detect_faces(frame)
    results = []
    
    for face_data in faces:
        # Tính embedding
        embedding = get_face_embedding(face_data["face_img"])
        
        # Nhận diện face
        face_id, similarity = identify_face(embedding, threshold)
        
        # Tính diện tích
        x1, y1, x2, y2 = face_data["box"]
        area = (x2 - x1) * (y2 - y1)
        
        results.append({
            "face_id": face_id,
            "similarity": similarity,
            "box": face_data["box"],
            "confidence": face_data["confidence"],
            "face_img": face_data["face_img"],
            "embedding": embedding,
            "area": area
        })
    
    return results


def get_largest_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Lấy khuôn mặt lớn nhất (theo diện tích bounding box).
    
    Args:
        faces: List faces từ detect_and_identify_faces()
    
    Returns:
        Face dict với diện tích lớn nhất, hoặc None nếu không có face
    """
    if not faces:
        return None
    
    return max(faces, key=lambda f: f["area"])


def draw_face_boxes(
    frame: np.ndarray,
    faces: List[Dict[str, Any]],
    selected_face_id: Optional[int] = None,
    show_emotion: bool = True,
    emotions: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Vẽ bounding box và thông tin lên frame.
    
    Args:
        frame: Frame ảnh gốc
        faces: List faces từ detect_and_identify_faces()
        selected_face_id: ID của face được chọn (sẽ highlight)
        show_emotion: Có hiển thị emotion không
        emotions: Dict {face_id: emotion_name} nếu muốn hiển thị emotion
    
    Returns:
        Frame đã được vẽ annotations
    """
    result_frame = frame.copy()
    
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        face_id = face["face_id"]
        
        # Màu: xanh lá cho face được chọn, xanh dương cho các face khác
        if selected_face_id is not None and face_id == selected_face_id:
            color = (0, 255, 0)  # Green
            thickness = 3
        else:
            color = (255, 0, 0)  # Blue
            thickness = 2
        
        # Vẽ bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Tạo label text
        label = f"ID:{face_id}"
        if show_emotion and emotions and face_id in emotions:
            label += f" - {emotions[face_id]}"
        
        # Vẽ label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            result_frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )
        
        # Vẽ text
        cv2.putText(
            result_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2
        )
    
    return result_frame


def reset_face_database():
    """Reset face database - xóa tất cả faces đã nhận diện (chỉ in-memory, không xóa DB)."""
    global _face_db, _next_face_id, _db_loaded
    _face_db = {}
    _next_face_id = 0
    _db_loaded = False  # Cho phép reload từ DB lần sau
    print("[FaceRecognition] In-memory face database reset.")


def get_face_count() -> int:
    """Trả về số lượng faces đã đăng ký trong database."""
    _load_face_db_from_database()  # Đảm bảo đã load
    return len(_face_db)


def reload_face_database():
    """Force reload face database từ SQLite."""
    global _db_loaded
    _db_loaded = False
    _load_face_db_from_database()
    print(f"[FaceRecognition] Reloaded. Total faces: {len(_face_db)}")


def get_face_db_info() -> Dict[str, Any]:
    """
    Lấy thông tin về face database hiện tại.
    
    Returns:
        Dict với thông tin: total_faces, next_id, face_ids, db_loaded
    """
    _load_face_db_from_database()
    return {
        "total_faces": len(_face_db),
        "next_id": _next_face_id,
        "face_ids": list(_face_db.keys()),
        "db_loaded": _db_loaded,
    }

