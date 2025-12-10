from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import sqlite3
import json
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "emotion_memory.db")


def _get_connection() -> sqlite3.Connection:
    """
    Lấy kết nối SQLite, đảm bảo DB và bảng đã tồn tại.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    # Tạo bảng với schema mới (bao gồm face features)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emotion_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            dominant_emotion TEXT NOT NULL,
            emotions_json TEXT NOT NULL,
            text TEXT NOT NULL,
            similarity REAL,
            face_embedding BLOB,
            box_json TEXT
        )
        """
    )
    conn.commit()
    
    # Migrate: thêm cột mới nếu chưa có (cho DB cũ)
    try:
        conn.execute("ALTER TABLE emotion_memory ADD COLUMN similarity REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Cột đã tồn tại
    
    try:
        conn.execute("ALTER TABLE emotion_memory ADD COLUMN face_embedding BLOB")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    
    try:
        conn.execute("ALTER TABLE emotion_memory ADD COLUMN box_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    
    return conn


def build_context_text_from_emotion(
    dominant_emotion: str,
    emotions: Dict[str, float],
) -> str:
    """
    Biến kết quả cảm xúc thành 1 đoạn text mô tả để lưu vào memory.
    """
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M")

    scores_str_parts = []
    for name, score in emotions.items():
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        scores_str_parts.append(f"{name}: {score_f:.2f}")
    scores_str = ", ".join(scores_str_parts)

    return (
        f"Thời điểm: {time_str}. "
        f"Cảm xúc chính của người dùng là {dominant_emotion}. "
        f"Tỉ lệ các cảm xúc: {scores_str}."
    )


def upsert_emotion_memory(
    user_id: str,
    dominant_emotion: str,
    emotions: Dict[str, float],
    similarity: Optional[float] = None,
    face_embedding: Optional[np.ndarray] = None,
    box: Optional[List[int]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Lưu 1 lần đo cảm xúc vào SQLite dưới dạng memory.
    
    Args:
        user_id: ID của người dùng (face_id)
        dominant_emotion: Cảm xúc chính
        emotions: Dict các cảm xúc và xác suất
        similarity: Độ tương đồng với face đã biết
        face_embedding: Vector đặc trưng khuôn mặt (512-D numpy array)
        box: Bounding box [x1, y1, x2, y2]
        extra_metadata: Metadata bổ sung (legacy support)
    
    Returns:
        Timestamp của record đã lưu
    """
    text = build_context_text_from_emotion(dominant_emotion, emotions)
    timestamp = datetime.now().isoformat()

    # Xử lý extra_metadata (legacy support)
    if extra_metadata:
        if similarity is None and "similarity" in extra_metadata:
            similarity = extra_metadata.get("similarity")
        if face_embedding is None and "face_embedding" in extra_metadata:
            face_embedding = extra_metadata.get("face_embedding")
        if box is None and "box" in extra_metadata:
            box = extra_metadata.get("box")

    # Convert face_embedding numpy array to bytes for BLOB storage
    embedding_blob = None
    if face_embedding is not None:
        if isinstance(face_embedding, np.ndarray):
            embedding_blob = face_embedding.tobytes()
        elif isinstance(face_embedding, bytes):
            embedding_blob = face_embedding

    # Convert box to JSON
    box_json = json.dumps(box) if box is not None else None

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO emotion_memory 
        (user_id, timestamp, dominant_emotion, emotions_json, text, similarity, face_embedding, box_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            timestamp,
            dominant_emotion,
            json.dumps(emotions),
            text,
            similarity,
            embedding_blob,
            box_json,
        ),
    )
    conn.commit()
    return timestamp


def search_emotion_memory(
    user_id: str,
    current_emotion: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Lấy lần đo cảm xúc gần nhất của user trong SQLite.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT id, user_id, timestamp, dominant_emotion, emotions_json, text, 
               similarity, face_embedding, box_json
        FROM emotion_memory
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (user_id, top_k),
    )

    rows = cursor.fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        rid, uid, ts, dom, emotions_json, text, sim, emb_blob, box_json = row
        
        try:
            emotions = json.loads(emotions_json)
        except Exception:
            emotions = {}

        # Convert BLOB back to numpy array (FaceNet outputs float32)
        face_embedding = None
        if emb_blob is not None:
            face_embedding = np.frombuffer(emb_blob, dtype=np.float32)

        # Parse box JSON
        box = None
        if box_json is not None:
            try:
                box = json.loads(box_json)
            except Exception:
                pass

        payload: Dict[str, Any] = {
            "user_id": uid,
            "timestamp": ts,
            "dominant_emotion": dom,
            "emotions": emotions,
            "text": text,
            "similarity": sim,
            "face_embedding": face_embedding,
            "box": box,
        }

        results.append(
            {
                "id": rid,
                "score": None,
                "text": text,
                "payload": payload,
            }
        )

    return results


def get_user_face_embedding(user_id: str) -> Optional[np.ndarray]:
    """
    Lấy face embedding gần nhất của user.
    Dùng để so sánh với face mới detect.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT face_embedding
        FROM emotion_memory
        WHERE user_id = ? AND face_embedding IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (user_id,),
    )
    
    row = cursor.fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32)
    return None


def get_all_user_embeddings() -> Dict[int, np.ndarray]:
    """
    Lấy tất cả face embeddings đã lưu, mỗi user_id lấy embedding mới nhất.
    Dùng để khởi tạo lại face_db khi restart app.
    
    Returns:
        Dict {user_id (int): face_embedding (np.ndarray)}
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT user_id, face_embedding, MAX(timestamp)
        FROM emotion_memory
        WHERE face_embedding IS NOT NULL
        GROUP BY user_id
        ORDER BY user_id
        """
    )
    
    result = {}
    for row in cursor.fetchall():
        user_id_str, emb_blob, _ = row
        if emb_blob:
            try:
                user_id = int(user_id_str)
                embedding = np.frombuffer(emb_blob, dtype=np.float32)
                result[user_id] = embedding
            except (ValueError, TypeError):
                pass
    
    return result
