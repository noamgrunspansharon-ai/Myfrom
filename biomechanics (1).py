import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import biomechanics

# --- UI Setup ---
st.set_page_config(page_title="T&F AI Coach", page_icon="🏃", layout="centered")
st.title("🏃‍♂️ Track & Field AI Coach")
st.write("Upload a side-on video to analyze your jump or hurdle mechanics.")

# --- App State & Inputs ---
event_choice = st.selectbox("Select Event", list(biomechanics.ANALYZERS.keys()))
athlete_height = st.number_input("Athlete Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

def process_video_to_keypoints(video_path):
    """Extracts (N, 33, 3) BlazePose keypoints in pixel coordinates."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    keypoints_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            frame_kps = np.zeros((33, 3))
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_kps[i] = [lm.x * width, lm.y * height, lm.visibility]
            keypoints_list.append(frame_kps)
        else:
            keypoints_list.append(np.full((33, 3), np.nan))
            
    cap.release()
    pose.close()
    
    return np.array(keypoints_list), fps, (width, height)

# --- Execution ---
if uploaded_file is not None:
    if st.button("Analyze Mechanics"):
        with st.spinner("Extracting kinematics... this may take a minute."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            try:
                keypoints, fps, frame_size = process_video_to_keypoints(tfile.name)
                
                if len(keypoints) < 10:
                    st.error("Video too short or unreadable.")
                else:
                    kin = biomechanics.build_kinematics(
                        keypoints=keypoints, 
                        fps=fps, 
                        frame_size=frame_size,
                        athlete_height_m=athlete_height
                    )
                    
                    result = biomechanics.analyze_event(event_choice, kin)
                    
                    st.success(f"Analysis Complete: {event_choice}")
                    
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                            
                    st.subheader("Key Metrics")
                    cols = st.columns(2)
                    for i, (metric, value) in enumerate(result.metrics.items()):
                        clean_name = metric.replace('_', ' ').title()
                        with cols[i % 2]:
                            st.metric(clean_name, f"{value:.2f}" if not np.isnan(value) else "N/A")
                    
                    st.subheader("Key Frames Identified")
                    for kf_name, frame_idx in result.key_frames.items():
                        st.write(f"**{kf_name}:** Frame {frame_idx}")
                        if frame_idx in result.annotations:
                            for note in result.annotations[frame_idx]:
                                st.caption(f"- {note}")
                                
                    if result.series:
                        st.subheader("Kinematic Charts")
                        for series_name, data in result.series.items():
                            st.line_chart(data, use_container_width=True)
                            
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
