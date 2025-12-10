import av
import cv2
import streamlit as st
import time
import numpy as np
from PIL import Image
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from services.deepface_service import predict_emotion
from services.face_recognition_service import (
    detect_and_identify_faces,
    get_largest_face,
    draw_face_boxes,
    reset_face_database,
    get_face_count,
)
# upsert_emotion_memory được gọi trong emotion_agent_service, không cần import ở đây
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
    Detect faces với YOLO và track face_id với FaceNet.
    """

    def __init__(self):
        # Frame BGR mới nhất từ camera
        self.last_frame_bgr = None
        # Frame đã được vẽ annotations (faces, emotions)
        self.annotated_frame = None
        # Ảnh đã được "chụp" (PIL Image) giống như Take photo
        self.captured_image = None
        # Kết quả phân tích cảm xúc gần nhất
        self.last_result = None
        # Danh sách faces được detect trong frame hiện tại
        self.detected_faces = []
        # Face được chọn (lớn nhất)
        self.selected_face = None
        # Dict lưu emotion cho từng face_id {face_id: emotion_name}
        self.face_emotions = {}
        # Enable/disable real-time detection trong recv()
        self.realtime_detection = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Lưu frame mới nhất
        img_bgr = frame.to_ndarray(format="bgr24")
        self.last_frame_bgr = img_bgr.copy()
        
        # Real-time face detection và annotation
        if self.realtime_detection:
            try:
                # Detect và identify faces
                faces = detect_and_identify_faces(img_bgr)
                self.detected_faces = faces
                
                # Chọn face lớn nhất
                largest_face = get_largest_face(faces)
                self.selected_face = largest_face
                
                # Predict emotion cho từng face và cập nhật dict
                for face in faces:
                    face_id = face["face_id"]
                    face_img = face["face_img"]
                    
                    # Predict emotion bằng MLT model
                    emotion, prob, emotion_probs = predict_emotion(face_img)
                    self.face_emotions[face_id] = emotion
                    
                    # Lưu vào face dict để dùng sau
                    face["emotion"] = emotion
                    face["emotion_prob"] = prob
                    face["emotion_probs"] = emotion_probs
                
                # Vẽ annotations lên frame
                selected_id = largest_face["face_id"] if largest_face else None
                annotated = draw_face_boxes(
                    img_bgr,
                    faces,
                    selected_face_id=selected_id,
                    show_emotion=True,
                    emotions=self.face_emotions
                )
                self.annotated_frame = annotated
                
                # Trả về frame đã annotate
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                
            except Exception as e:
                # Nếu lỗi, trả về frame gốc
                pass
        
        return frame


def render_camera_auto(interval_seconds: int = 15):
    """
    Giao diện và logic cho chế độ Camera auto.
    Sequential flow: Detect emotion → Call Gemini → Show response → Detect tiếp
    Tích hợp YOLO face detection và FaceNet face recognition.
    """
    st.subheader("🍽️ Hệ thống hỗ trợ phục vụ khách hàng - Nhà hàng")
    st.write(
        "**Quy trình:** Detect khuôn mặt khách → Nhận diện (Face ID) → Phân tích cảm xúc → AI đưa hướng dẫn cho nhân viên\n\n"
        "Bật camera để theo dõi cảm xúc khách hàng và nhận hướng dẫn phục vụ phù hợp."
    )
    
    # Khởi tạo session state
    if "previous_emotion" not in st.session_state:
        st.session_state.previous_emotion = {}  # Dict {face_id: emotion} thay vì single value
    if "is_gemini_processing" not in st.session_state:
        st.session_state.is_gemini_processing = False
    if "last_gemini_suggestion" not in st.session_state:
        st.session_state.last_gemini_suggestion = {}  # Dict {face_id: suggestion}
    if "last_detection_time" not in st.session_state:
        st.session_state.last_detection_time = 0
    if "waiting_for_ai" not in st.session_state:
        st.session_state.waiting_for_ai = False
    if "is_playing_audio" not in st.session_state:
        st.session_state.is_playing_audio = False
    if "current_audio_file" not in st.session_state:
        st.session_state.current_audio_file = None
    if "current_face_id" not in st.session_state:
        st.session_state.current_face_id = None
    
    # Style cho nút/checkbox (font 16px)
    st.markdown(
        """
        <style>
        .stMarkdown { font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Control buttons
    col1, col2, col3, col4 = st.columns([5, 5, 5, 5], gap="small")
    with col1:
        if st.button("🔄 Reset và bắt đầu lại", use_container_width=True):
            st.session_state.previous_emotion = {}
            st.session_state.last_gemini_suggestion = {}
            st.session_state.waiting_for_ai = False
            st.session_state.is_gemini_processing = False
            st.session_state.last_detection_time = 0
            st.session_state.force_detect = True
            st.session_state.current_face_id = None
            # Reset face database
            reset_face_database()
            st.success("✅ Đã reset! Sẵn sàng detect cảm xúc mới.")
            st.rerun()
    
    with col2:
        if st.button("▶️ Detect cảm xúc ngay", use_container_width=True):
            st.session_state.waiting_for_ai = False
            st.session_state.last_detection_time = 0
            st.session_state.force_detect = True
            st.rerun()
    
    with col3:
        auto_mode = st.checkbox("🔄 Tự động detect", value=False, key="auto_detect_mode")
    
    with col4:
        # Hiển thị số face đã nhận diện
        face_count = get_face_count()
        st.metric("👥 Faces", face_count)
    
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
        async_processing=True,
    )

    # Placeholders cho UI
    face_info_placeholder = st.empty()
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
            current_face_id = st.session_state.current_face_id
            if current_face_id is not None and processor.last_result:
                result = processor.last_result
                dominant_emotion = result.get("dominant_emotion")
                emotions = result.get("emotion", {})
                face_info_placeholder.info(f"👤 **Face ID:** {current_face_id}")
                result_placeholder.success(f"**Cảm xúc chính**: {dominant_emotion}")
                if emotions:
                    chart_placeholder.subheader("Chi tiết các cảm xúc")
                    chart_placeholder.bar_chart(emotions)
            if current_face_id is not None and current_face_id in st.session_state.last_gemini_suggestion:
                suggestion_placeholder.markdown(
                    f"### 🍽️ Hướng dẫn phục vụ khách hàng\n\n{st.session_state.last_gemini_suggestion[current_face_id]}"
                )
        else:
            # Kiểm tra xem có nên detect không
            should_detect = (
                st.session_state.force_detect or
                auto_mode or 
                st.session_state.last_detection_time == 0 or
                (time.time() - st.session_state.last_detection_time) > interval_seconds
            )
            
            # Reset force_detect flag sau khi dùng
            if st.session_state.force_detect:
                st.session_state.force_detect = False
            
            # Process face detection results
            if should_detect and processor.selected_face is not None:
                try:
                    selected_face = processor.selected_face
                    face_id = selected_face["face_id"]
                    st.session_state.current_face_id = face_id
                    
                    # Lấy thông tin emotion từ processor
                    dominant_emotion = selected_face.get("emotion", "Unknown")
                    emotion_probs = selected_face.get("emotion_probs", {})
                    
                    # Lưu captured image
                    if processor.annotated_frame is not None:
                        # Chuyển BGR -> RGB cho PIL
                        frame_rgb = cv2.cvtColor(processor.annotated_frame, cv2.COLOR_BGR2RGB)
                        processor.captured_image = Image.fromarray(frame_rgb)
                    
                    # Tạo result dict (bao gồm cả face features)
                    result = {
                        "face_id": face_id,
                        "dominant_emotion": dominant_emotion,
                        "emotion": emotion_probs,
                        "similarity": selected_face.get("similarity", 0),
                        "box": selected_face.get("box", []),
                        "face_embedding": selected_face.get("embedding"),  # 512-D vector
                    }
                    processor.last_result = result
                    
                    # Lưu emotion vào DB được thực hiện trong emotion_agent_service
                    # khi gọi generate_advice_with_memory_from_result()
                    
                    st.session_state.last_detection_time = time.time()
                    
                except Exception as e:
                    st.warning(f"Lỗi khi detect cảm xúc: {e}")

        # Hiển thị ảnh nếu có
        if processor.captured_image is not None:
            st.image(
                processor.captured_image,
                caption="Ảnh đã capture (với face annotations)",
                use_container_width=True,
            )

        # Xử lý kết quả và gọi Gemini
        if not st.session_state.is_gemini_processing and not st.session_state.waiting_for_ai:
            result = processor.last_result
            if result:
                face_id = result.get("face_id")
                dominant_emotion = result.get("dominant_emotion")
                emotions = result.get("emotion", {})
                similarity = result.get("similarity", 0)

                face_info_placeholder.info(
                    f"👤 **Face ID:** {face_id} | "
                    f"📊 **Similarity:** {similarity:.2f}"
                )
                result_placeholder.success(f"**Cảm xúc chính**: {dominant_emotion}")

                if emotions:
                    chart_placeholder.subheader("Chi tiết các cảm xúc")
                    chart_placeholder.bar_chart(emotions)

                # --- Gọi Gemini CHỈ KHI emotion thay đổi cho face_id đó ---
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
                        # Lấy emotion trước đó của face_id này
                        previous_emotion = st.session_state.previous_emotion.get(face_id)
                        emotion_changed = previous_emotion != dominant_emotion

                        if emotion_changed or previous_emotion is None:
                            # Set flags
                            st.session_state.is_gemini_processing = True
                            st.session_state.waiting_for_ai = True

                            # Gọi Gemini với face_id làm user_id, lưu cả face features vào DB
                            with st.spinner(f"🤔 AI đang phân tích cảm xúc '{dominant_emotion}' cho Face ID {face_id}..."):
                                try:
                                    suggestion_text = generate_advice_with_memory_from_result(
                                        model=model,
                                        dominant_emotion=dominant_emotion,
                                        emotions=emotions,
                                        user_id=str(face_id),
                                        similarity=result.get("similarity"),
                                        face_embedding=result.get("face_embedding"),
                                        box=result.get("box"),
                                    )
                                except Exception as e:
                                    suggestion_text = f"⚠️ Lỗi khi gọi Gemini: {str(e)}"
                                    st.error(f"❌ Exception: {e}")

                            # Clear flags sau khi xong
                            st.session_state.is_gemini_processing = False
                            st.session_state.waiting_for_ai = False

                            # Lưu emotion và suggestion theo face_id
                            st.session_state.previous_emotion[face_id] = dominant_emotion
                            st.session_state.last_gemini_suggestion[face_id] = suggestion_text

                            # Hiển thị response
                            if suggestion_text and suggestion_text.strip():
                                if suggestion_text.startswith("⚠️"):
                                    suggestion_placeholder.warning(suggestion_text)
                                    status_placeholder.warning("⚠️ Có lỗi xảy ra khi gọi AI")
                                else:
                                    suggestion_placeholder.markdown(
                                        f"### 🍽️ Hướng dẫn phục vụ - Khách #{face_id}\n\n{suggestion_text}"
                                    )
                                    
                                    # TTS
                                    if tts_enabled:
                                        st.session_state.is_playing_audio = True
                                        
                                        emotion_intro = create_emotion_intro(dominant_emotion)
                                        full_text_to_speak = emotion_intro + suggestion_text
                                        
                                        with st.spinner("🔊 Đang tạo audio..."):
                                            audio_file = text_to_speech_file(full_text_to_speak, lang="vi", slow=False)
                                        
                                        if audio_file:
                                            st.session_state.current_audio_file = audio_file
                                            audio_placeholder.audio(audio_file, format="audio/mp3", autoplay=True)
                                            
                                            estimated_duration = estimate_speech_duration(full_text_to_speak)
                                            status_placeholder.info(
                                                f"🔊 **Đang phát audio...** (ước tính ~{int(estimated_duration)}s). "
                                                "Sau khi phát xong sẽ detect cảm xúc tiếp theo."
                                            )
                                            
                                            time.sleep(estimated_duration + 1)
                                            
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
                            
                            # Auto mode - detect tiếp
                            if auto_mode and suggestion_text and not suggestion_text.startswith("⚠️"):
                                if not tts_enabled or not st.session_state.is_playing_audio:
                                    time.sleep(1)
                                    st.rerun()
                        else:
                            # Emotion không đổi cho face_id này
                            if face_id in st.session_state.last_gemini_suggestion:
                                suggestion_placeholder.markdown(
                                    f"### 🍽️ Hướng dẫn phục vụ - Khách #{face_id}\n\n{st.session_state.last_gemini_suggestion[face_id]}"
                                )
                                status_placeholder.info(
                                    f"ℹ️ Cảm xúc '{dominant_emotion}' của Face ID {face_id} không thay đổi. "
                                    "Đang tiếp tục detect..."
                                )
                            else:
                                suggestion_placeholder.info(
                                    f"ℹ️ Cảm xúc '{dominant_emotion}' của Face ID {face_id}. "
                                    "Đang tiếp tục detect..."
                                )
                            
                            st.session_state.previous_emotion[face_id] = dominant_emotion
                            
                            if auto_mode:
                                st.session_state.last_detection_time = time.time() - interval_seconds + 1
                                time.sleep(1)
                                st.rerun()
            else:
                if processor.last_frame_bgr is None:
                    result_placeholder.info(
                        "📷 **Đang chờ camera...**\n\n"
                        "Hãy đảm bảo camera đã được bật và cho phép truy cập."
                    )
                elif processor.selected_face is None:
                    result_placeholder.info(
                        "👤 **Không phát hiện khuôn mặt**\n\n"
                        "Hãy đưa khuôn mặt vào camera để bắt đầu detect."
                    )
                else:
                    result_placeholder.info("💡 Nhấn 'Detect cảm xúc ngay' để bắt đầu phân tích.")
