import time

import av
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from services.deepface_service import analyze_emotion


class EmotionVideoProcessor(VideoProcessorBase):
    """
    Video processor dùng cho streamlit-webrtc.
    Chỉ giữ frame mới nhất từ camera, để thread chính quyết định khi nào chụp.
    """

    def __init__(self):
        # Frame BGR mới nhất từ camera
        self.last_frame_bgr = None
        # Ảnh đã được "chụp" (PIL Image) giống như Take photo
        self.captured_image = None
        # Kết quả phân tích cảm xúc gần nhất
        self.last_result = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Lưu frame mới nhất, không phân tích tại đây
        img_bgr = frame.to_ndarray(format="bgr24")
        self.last_frame_bgr = img_bgr
        return frame


def render_camera_auto(interval_seconds: int = 15):
    """
    Giao diện và logic cho chế độ Camera auto.
    Mỗi `interval_seconds` sẽ tự động chụp 1 lần và phân tích cảm xúc.
    """
    st.subheader(f"Camera tự động chụp ảnh mỗi {interval_seconds} giây")
    st.write(
        f"Bật camera bên dưới, hệ thống sẽ tự động **\"Take photo\" mỗi {interval_seconds} giây** "
        "và phân tích cảm xúc cho bạn, giống như bạn nhấn nút chụp tay."
    )

    # Tự động refresh toàn bộ trang đúng theo interval
    st_autorefresh(
        interval=interval_seconds * 1000,
        key=f"camera-auto-{interval_seconds}-refresh",
    )

    webrtc_ctx = webrtc_streamer(
        key=f"emotion-auto-{interval_seconds}",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=EmotionVideoProcessor,
    )

    result_placeholder = st.empty()
    chart_placeholder = st.empty()

    if webrtc_ctx.video_processor is not None:
        processor = webrtc_ctx.video_processor

        # Mỗi lần trang reload (interval giây), nếu đã có frame thì coi như vừa "Take photo"
        if processor.last_frame_bgr is not None:
            try:
                frame_rgb = processor.last_frame_bgr[:, :, ::-1]  # BGR -> RGB
                image = Image.fromarray(frame_rgb)
                processor.captured_image = image

                # Phân tích cảm xúc giống như nhấn Take photo
                with st.spinner(f"Đang phân tích cảm xúc (auto {interval_seconds}s)..."):
                    result = analyze_emotion(image)
                processor.last_result = result
            except Exception as e:
                st.warning(f"Lỗi khi auto capture: {e}")

        # Nếu đã có ảnh được chụp, hiển thị giống camera chụp tay
        if processor.captured_image is not None:
            st.image(
                processor.captured_image,
                caption=f"Ảnh auto capture (mỗi ~{interval_seconds}s, giống Take photo)",
                use_column_width=True,
            )

        result = processor.last_result
        if result:
            dominant_emotion = result.get("dominant_emotion")
            emotions = result.get("emotion", {})

            result_placeholder.success(
                f"**Cảm xúc chính (cập nhật mỗi ~{interval_seconds}s)**: {dominant_emotion}"
            )

            if emotions:
                chart_placeholder.subheader("Chi tiết các cảm xúc")
                chart_placeholder.bar_chart(emotions)
        else:
            result_placeholder.info(
                "Đang chờ frame đầu tiên được phân tích... "
                "Hãy đảm bảo camera đã được bật và cho phép truy cập."
            )


