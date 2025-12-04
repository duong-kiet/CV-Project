import av
import streamlit as st
import time
from PIL import Image
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from services.deepface_service import analyze_emotion
from services.gemini_service import (
    get_gemini_api_key,
    init_gemini,
    create_emotion_intro,
)
from services.emotion_agent_service import (
    generate_advice_with_memory_from_result,
)

from services.tts_service import text_to_speech_file, estimate_speech_duration, cleanup_audio_file


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
    Sequential flow: Detect emotion → Call Gemini → Show response → Detect tiếp
    """
    st.subheader("🤖 Trợ lý cảm xúc AI - Chế độ tự động")
    st.write(
        "**Quy trình:** Detect cảm xúc → AI phân tích và đưa lời động viên → Detect tiếp\n\n"
        "Bật camera bên dưới, hệ thống sẽ tự động detect cảm xúc và đợi AI trả lời xong mới detect tiếp."
    )
    
    # Khởi tạo session state
    if "previous_emotion" not in st.session_state:
        st.session_state.previous_emotion = None
    if "is_gemini_processing" not in st.session_state:
        st.session_state.is_gemini_processing = False
    if "last_gemini_suggestion" not in st.session_state:
        st.session_state.last_gemini_suggestion = None
    if "last_detection_time" not in st.session_state:
        st.session_state.last_detection_time = 0
    if "waiting_for_ai" not in st.session_state:
        st.session_state.waiting_for_ai = False
    if "is_playing_audio" not in st.session_state:
        st.session_state.is_playing_audio = False
    if "current_audio_file" not in st.session_state:
        st.session_state.current_audio_file = None
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset và bắt đầu lại"):
            st.session_state.previous_emotion = None
            st.session_state.last_gemini_suggestion = None
            st.session_state.waiting_for_ai = False
            st.session_state.is_gemini_processing = False
            st.session_state.last_detection_time = 0
            st.session_state.force_detect = True
            st.success("✅ Đã reset! Sẵn sàng detect cảm xúc mới.")
            st.rerun()
    
    with col2:
        if st.button("▶️ Detect cảm xúc ngay"):
            st.session_state.waiting_for_ai = False
            st.session_state.last_detection_time = 0
            st.session_state.force_detect = True
            st.rerun()
    
    with col3:
        auto_mode = st.checkbox("🔄 Tự động detect", value=False, key="auto_detect_mode")
    
    # TTS settings
    tts_enabled = st.checkbox("🔊 Bật đọc text-to-speech", value=True, key="tts_enabled")
    
    # Khởi tạo force_detect flag
    if "force_detect" not in st.session_state:
        st.session_state.force_detect = False

    webrtc_ctx = webrtc_streamer(
        key=f"emotion-auto-{interval_seconds}",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=EmotionVideoProcessor,
    )

    result_placeholder = st.empty()
    chart_placeholder = st.empty()
    suggestion_placeholder = st.empty()
    status_placeholder = st.empty()
    audio_placeholder = st.empty()

    if webrtc_ctx.video_processor is not None:
        processor = webrtc_ctx.video_processor

        # Nếu đang chờ AI response, chỉ hiển thị kết quả cũ
        if st.session_state.is_gemini_processing or st.session_state.waiting_for_ai:
            status_placeholder.info("⏳ **Đang chờ AI trả lời...** Vui lòng đợi.")
            # Hiển thị kết quả cũ nếu có
            if processor.last_result:
                result = processor.last_result
                dominant_emotion = result.get("dominant_emotion")
                emotions = result.get("emotion", {})
                result_placeholder.success(f"**Cảm xúc chính**: {dominant_emotion}")
                if emotions:
                    chart_placeholder.subheader("Chi tiết các cảm xúc")
                    chart_placeholder.bar_chart(emotions)
            if st.session_state.last_gemini_suggestion:
                suggestion_placeholder.markdown(
                    f"### 💬 Gợi ý từ trợ lý cảm xúc\n\n{st.session_state.last_gemini_suggestion}"
                )
        else:
            # Kiểm tra xem có nên detect không (auto mode, button được nhấn, hoặc force detect)
            should_detect = (
                st.session_state.force_detect or
                auto_mode or 
                st.session_state.last_detection_time == 0 or
                (time.time() - st.session_state.last_detection_time) > interval_seconds
            )
            
            # Reset force_detect flag sau khi dùng
            if st.session_state.force_detect:
                st.session_state.force_detect = False
            
            # Nếu không đang chờ AI và có frame, luôn capture và detect
            if should_detect and processor.last_frame_bgr is not None:
                try:
                    frame_rgb = processor.last_frame_bgr[:, :, ::-1]  # BGR -> RGB
                    image = Image.fromarray(frame_rgb)
                    processor.captured_image = image

                    # Phân tích cảm xúc
                    with st.spinner("🔍 Đang phân tích cảm xúc..."):
                        result = analyze_emotion(image)
                    processor.last_result = result
                    st.session_state.last_detection_time = time.time()
                except Exception as e:
                    st.warning(f"Lỗi khi detect cảm xúc: {e}")

        # Hiển thị ảnh nếu có
        if processor.captured_image is not None:
            st.image(
                processor.captured_image,
                caption="Ảnh đã capture",
                use_column_width=True,
            )

        # Xử lý kết quả và gọi Gemini
        if not st.session_state.is_gemini_processing and not st.session_state.waiting_for_ai:
            result = processor.last_result
            if result:
                dominant_emotion = result.get("dominant_emotion")
                emotions = result.get("emotion", {})

                result_placeholder.success(f"**Cảm xúc chính**: {dominant_emotion}")

                if emotions:
                    chart_placeholder.subheader("Chi tiết các cảm xúc")
                    chart_placeholder.bar_chart(emotions)

                # --- Gọi Gemini CHỈ KHI emotion thay đổi ---
                api_key = get_gemini_api_key()
                if not api_key:
                    suggestion_placeholder.warning(
                        "⚠️ **Chưa tìm thấy Gemini API key!**\n\n"
                        "Vui lòng:\n"
                        "1. Tạo file `.env` trong thư mục gốc với nội dung: `GEMINI_API_KEY=your_key_here`\n"
                        "2. Hoặc nhập API key ở sidebar\n"
                        "3. Hoặc set biến môi trường: `export GEMINI_API_KEY=your_key_here`"
                    )
                else:
                    # Khởi tạo model một lần / session
                    if "gemini_model" not in st.session_state:
                        with st.spinner("Đang khởi tạo Gemini model..."):
                            model, model_info = init_gemini(api_key)
                        if model is None:
                            suggestion_placeholder.error(f"❌ **Lỗi khởi tạo Gemini:** {model_info}")
                        else:
                            st.session_state.gemini_model = model
                            st.session_state.gemini_model_name = model_info

                    model = st.session_state.get("gemini_model")
                    if model:
                        # CHỈ gọi Gemini khi emotion thay đổi
                        previous_emotion = st.session_state.previous_emotion
                        emotion_changed = previous_emotion != dominant_emotion

                        if emotion_changed or previous_emotion is None:
                            # Set flags
                            st.session_state.is_gemini_processing = True
                            st.session_state.waiting_for_ai = True

                            # Gọi Gemini + SQLite (memory) và đợi response (blocking) với spinner
                            with st.spinner(f"🤔 AI trợ lý cảm xúc đang suy nghĩ về cảm xúc '{dominant_emotion}'..."):
                                try:
                                    # Sử dụng agent có memory (SQLite) thay vì chỉ cảm xúc hiện tại
                                    suggestion_text = generate_advice_with_memory_from_result(
                                        model=model,
                                        dominant_emotion=dominant_emotion,
                                        emotions=emotions,
                                        user_id="default_user",
                                    )
                                except Exception as e:
                                    suggestion_text = f"⚠️ Lỗi khi gọi Gemini: {str(e)}"
                                    st.error(f"❌ Exception: {e}")

                            # Clear flags sau khi xong
                            st.session_state.is_gemini_processing = False
                            st.session_state.waiting_for_ai = False

                            # Lưu emotion và suggestion
                            st.session_state.previous_emotion = dominant_emotion
                            st.session_state.last_gemini_suggestion = suggestion_text

                            # Hiển thị response ngay lập tức
                            if suggestion_text and suggestion_text.strip():
                                if suggestion_text.startswith("⚠️"):
                                    suggestion_placeholder.warning(suggestion_text)
                                    status_placeholder.warning("⚠️ Có lỗi xảy ra khi gọi AI")
                                else:
                                    suggestion_placeholder.markdown(
                                        f"### 💬 Gợi ý từ trợ lý cảm xúc\n\n{suggestion_text}"
                                    )
                                    
                                    # Tạo và phát audio nếu TTS được bật
                                    if tts_enabled:
                                        st.session_state.is_playing_audio = True
                                        
                                        # Tạo câu giới thiệu cảm xúc
                                        emotion_intro = create_emotion_intro(dominant_emotion)
                                        
                                        # Nối câu giới thiệu với response từ AI
                                        full_text_to_speak = emotion_intro + suggestion_text
                                        
                                        # Tạo audio file
                                        with st.spinner("🔊 Đang tạo audio..."):
                                            audio_file = text_to_speech_file(full_text_to_speak, lang="vi", slow=False)
                                        
                                        if audio_file:
                                            st.session_state.current_audio_file = audio_file
                                            
                                            # Phát audio trong Streamlit
                                            audio_placeholder.audio(audio_file, format="audio/mp3", autoplay=True)
                                            
                                            # Ước tính thời gian (bao gồm cả intro)
                                            estimated_duration = estimate_speech_duration(full_text_to_speak)
                                            status_placeholder.info(
                                                f"🔊 **Đang phát audio...** (ước tính ~{int(estimated_duration)}s). "
                                                "Sau khi phát xong sẽ detect cảm xúc tiếp theo."
                                            )
                                            
                                            # Đợi audio phát xong (ước tính)
                                            time.sleep(estimated_duration + 1)  # +1s buffer
                                            
                                            # Cleanup audio file
                                            cleanup_audio_file(audio_file)
                                            st.session_state.is_playing_audio = False
                                            st.session_state.current_audio_file = None
                                            
                                            status_placeholder.success("✅ **Đã đọc xong!** Sẵn sàng detect cảm xúc tiếp theo.")
                                        else:
                                            st.session_state.is_playing_audio = False
                                            status_placeholder.warning("⚠️ Không thể tạo audio. Tiếp tục detect cảm xúc...")
                                    else:
                                        status_placeholder.success("✅ **AI đã trả lời xong!** Sẵn sàng detect cảm xúc tiếp theo.")
                            else:
                                suggestion_placeholder.error(
                                    f"❌ **Không nhận được phản hồi từ Gemini!**\n\n"
                                    f"**Debug:** suggestion_text = `{repr(suggestion_text)}`"
                                )
                                status_placeholder.error("❌ Không nhận được response từ AI")
                            
                            # Nếu auto mode, tự động detect tiếp sau khi có response và audio phát xong
                            if auto_mode and suggestion_text and not suggestion_text.startswith("⚠️"):
                                if not tts_enabled or not st.session_state.is_playing_audio:
                                    # Nếu không có TTS hoặc audio đã phát xong, đợi một chút rồi detect tiếp
                                    time.sleep(1)  # Đợi 1s để user đọc response
                                    st.rerun()
                        else:
                            # Emotion không đổi, hiển thị suggestion cũ nhưng vẫn tiếp tục detect
                            if st.session_state.last_gemini_suggestion:
                                suggestion_placeholder.markdown(
                                    f"### 💬 Gợi ý từ trợ lý cảm xúc\n\n{st.session_state.last_gemini_suggestion}"
                                )
                                status_placeholder.info(
                                    f"ℹ️ Cảm xúc '{dominant_emotion}' không thay đổi (giống '{previous_emotion}'). "
                                    "Đang tiếp tục capture và detect cảm xúc mới..."
                                )
                            else:
                                suggestion_placeholder.info(
                                    f"ℹ️ Cảm xúc '{dominant_emotion}' không thay đổi. "
                                    "Đang tiếp tục capture và detect cảm xúc mới..."
                                )
                            
                            # Vẫn cập nhật previous_emotion
                            st.session_state.previous_emotion = dominant_emotion
                            
                            # Nếu auto mode, trigger detect tiếp sau interval_seconds
                            if auto_mode:
                                # Set thời gian để detect tiếp
                                st.session_state.last_detection_time = time.time() - interval_seconds + 1
                                # Tự động rerun sau một chút để detect tiếp
                                time.sleep(1)
                                st.rerun()
            else:
                if processor.last_frame_bgr is None:
                    result_placeholder.info(
                        "📷 **Đang chờ camera...**\n\n"
                        "Hãy đảm bảo camera đã được bật và cho phép truy cập."
                    )
                else:
                    result_placeholder.info("💡 Nhấn 'Detect cảm xúc ngay' để bắt đầu phân tích.")


