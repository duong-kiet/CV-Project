from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "emotion_memory.db")


def _get_connection() -> sqlite3.Connection:
    """
    Lấy kết nối SQLite, đảm bảo DB và bảng đã tồn tại.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emotion_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            dominant_emotion TEXT NOT NULL,
            emotions_json TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.commit()
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
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Lưu 1 lần đo cảm xúc vào SQLite dưới dạng memory.
    """
    text = build_context_text_from_emotion(dominant_emotion, emotions)

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "text": text,
        "dominant_emotion": dominant_emotion,
        "emotions": emotions,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO emotion_memory (user_id, timestamp, dominant_emotion, emotions_json, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            payload["timestamp"],
            dominant_emotion,
            json.dumps(emotions),
            text,
        ),
    )
    conn.commit()
    return payload["timestamp"]


def search_emotion_memory(
    user_id: str,
    current_emotion: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Lấy lần đo cảm xúc gần nhất của user trong SQLite.
    - current_emotion được giữ lại chỉ để tương thích, KHÔNG dùng trong query.
    - Luôn trả về tối đa 1 phần tử (trong list) – cảm xúc mới nhất nếu có.
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT id, user_id, timestamp, dominant_emotion, emotions_json, text
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
        rid, uid, ts, dom, emotions_json, text = row
        try:
            emotions = json.loads(emotions_json)
        except Exception:
            emotions = {}

        payload: Dict[str, Any] = {
            "user_id": uid,
            "timestamp": ts,
            "dominant_emotion": dom,
            "emotions": emotions,
            "text": text,
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

