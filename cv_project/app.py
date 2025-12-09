import streamlit as st

from components.camera_auto import render_camera_auto
from components.camera_manual import render_camera_manual
from components.upload_image import render_upload_image
# from services.gemini_service import get_gemini_api_key

from services.gemini_service import get_gemini_api_key


st.set_page_config(
    page_title="Real-time Emotion Detection",
    page_icon="😄",
    layout="centered",
)


def main():
    st.title("Real-time Emotion Detection trên Web")
    st.write(
        "Ứng dụng demo sử dụng **DeepFace** để detect emotion từ camera hoặc ảnh upload. "
        "Do giới hạn của Streamlit, camera hoạt động theo kiểu chụp từng frame (gần real-time), "
        "không phải video stream liên tục như ứng dụng desktop."
    )

    # --- Cấu hình AI trợ lý cảm xúc (Gemini) ---
    st.sidebar.header("Cấu hình AI trợ lý cảm xúc (Gemini)")
    st.sidebar.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key",
        help="Dán API key từ Google AI Studio. Có thể đặt trong environment/secrets để không phải nhập lại.",
    )

    api_key = get_gemini_api_key()
    if api_key:
        st.sidebar.success("Đã sẵn sàng dùng Gemini cho trợ lý cảm xúc.")
    else:
        st.sidebar.info("Chưa có GEMINI_API_KEY, tính năng trợ lý cảm xúc sẽ bị tắt.")

    # --- Chọn nguồn input ---
    st.sidebar.header("Tùy chọn input")
    input_mode = st.sidebar.radio(
        "Nguồn ảnh",
        options=[
            "Camera auto 15s",
            "Camera chụp tay",
            "Upload ảnh",
        ],
        index=0,
    )

    st.markdown("---")

    if input_mode == "Camera auto 15s":
        render_camera_auto(interval_seconds=15)
    elif input_mode == "Camera chụp tay":
        render_camera_manual()
    else:
        render_upload_image()


if __name__ == "__main__":
    main()

