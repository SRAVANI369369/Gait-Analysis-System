import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import numpy as np
import pandas as pd

st.set_page_config(page_title="Gait Analysis System - Final Submission", layout="wide")

st.title("🏃 Gait Analysis System for Rehabilitation Monitoring")
st.subheader("Final Production Workspace - Full Kinematic Engine")
st.markdown("Markerless assessment platform with continuous data logging and automated reporting.")

# Mathematical helper function to calculate joint angle using the Law of Cosines
def calculate_joint_angle(p_hip, p_knee, p_ankle):
    v1 = np.array([p_hip[0] - p_knee[0], p_hip[1] - p_knee[1]])
    v2 = np.array([p_ankle[0] - p_knee[0], p_ankle[1] - p_knee[1]])
    
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    cos_theta = dot_product / (mag1 * mag2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cos_theta))
    return angle

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-pose.pt')

model = load_yolo_model()

uploaded_file = st.file_uploader("Upload your walking video clip...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 40/60 Split column viewport container
    col1, col2 = st.columns([4, 6])
    
    with col1:
        st.write("📺 **Pose Stream**")
        frame_placeholder = st.empty()
        
    with col2:
        metrics_placeholder = st.empty()
        
    # Initialization of tracking buffers and data logging arrays
    historical_ankle_x = []
    step_count = 0
    stride_lengths = []
    
    # NEW FOR FINAL SUBMISSION: Lists to accumulate time-series metrics for exporting
    timestamps_log = []
    angles_log = []
    cadence_log = []
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) > 16:
                r_hip = keypoints[12]   
                r_knee = keypoints[14]  
                r_ankle = keypoints[16] 
                
                # 1. Real-time Knee Angle Calculation
                knee_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                
                # 2. Step and Stride Length Tracking Logic
                historical_ankle_x.append(r_ankle[0])
                if len(historical_ankle_x) > 3:
                    if historical_ankle_x[-2] > historical_ankle_x[-3] and historical_ankle_x[-2] > historical_ankle_x[-1]:
                        step_count += 1
                        stride = abs(historical_ankle_x[-1] - historical_ankle_x[-3])
                        if stride > 10: 
                            stride_lengths.append(stride)
                
                # 3. Temporal Parameter Extraction: Cadence Calculation
                current_stride_avg = np.mean(stride_lengths) if stride_lengths else 0.0
                total_elapsed_time = frame_count / fps if fps > 0 else 0.03
                estimated_cadence = (step_count / total_elapsed_time) * 60 if total_elapsed_time > 0 else 0.0
                
                # Logging data metrics synchronously for reporting
                timestamps_log.append(round(total_elapsed_time, 2))
                angles_log.append(round(knee_angle, 1))
                cadence_log.append(round(estimated_cadence, 1))
                
                # Display metrics live inside Column 2
                metrics_placeholder.markdown(
                    f"""
                    ### 📊 Calculated Gait Parameters
                    
                    * **🎯 Live Knee Joint Angle:** `{knee_angle:.1f}°`
                    * **📏 Total Steps Counted:** `{step_count} steps`
                    * **📈 Average Stride Length:** `{current_stride_avg:.2f} pixels`
                    * **⏱️ Estimated Cadence:** `{estimated_cadence:.1f} steps/min`
                    
                    ---
                    **System Status:** Running Inference Pass...  
                    **Frame Time:** `{total_elapsed_time:.2f}s` | **Frame Index:** `{frame_count}`
                    """
                )
        
        frame_placeholder.image(image_rgb, channels="RGB", use_container_width=False, width=320)
        time.sleep(1/fps if fps > 0 else 0.03)
        
    cap.release()
    st.success("🎉 Gait parameter assessment processing complete!")
    
    # --- NEW FOR FINAL SUBMISSION: AUTOMATED REPORT GENERATION ENGINE ---
    st.markdown("### 📥 Patient Assessment Summary Report")
    
    if timestamps_log:
        # Create a unified structured DataFrame representing the session dataset
        report_df = pd.DataFrame({
            "Timestamp (Seconds)": timestamps_log,
            "Knee Flexion Angle (Degrees)": angles_log,
            "Estimated Cadence (Steps/Min)": cadence_log
        })
        
        # Display an summary performance card overview
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Peak Knee Flexion", value=f"{max(angles_log)}°")
        c2.metric(label="Total Steps Taken", value=f"{step_count} Steps")
        c3.metric(label="Average Session Cadence", value=f"{round(np.mean(cadence_log), 1)} SPM")
        
        # Render data export interaction matrix
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="💾 Download Patient Gait Analysis Report (.CSV)",
            data=csv_data,
            file_name="Patient_Gait_Analysis_Report.csv",
            mime="text/csv",
            use_container_width=True
        )
