import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ==============================================================================
# 1. PAGE LAYOUT & INTERFACE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Crop & Weed Analyzer",
    page_icon="🌱",
    layout="centered"
)

# Application Heading Banner
st.title("🌱 Automated Crop and Weed Target Identification")
st.markdown("""
This system leverages a trained computer vision model to distinguish between commercial **Crops** and competing **Weeds**. 
Upload field capture images below to process automated targeted spraying coordinates.
""")
st.write("---")

# ==============================================================================
# 2. CACHED MODEL INITIALIZATION FUNCTION
# ==============================================================================
@st.cache_resource
def load_detection_model():
    """
    Safely loads the trained weights file. Falls back to base yolov8n weights 
    with a helpful user warning if the project training cycles haven't run yet.
    """
    primary_weights = "best.pt"
    backup_weights = "yolov8n.pt"
    
    if os.path.exists(primary_weights):
        return YOLO(primary_weights), True
    else:
        # Fallback tracking if custom model weights are missing
        return YOLO(backup_weights), False

# Execute model loading hook
model, is_custom_model = load_detection_model()

# Display current core model deployment feedback banner
if is_custom_model:
    st.sidebar.success(" Custom Crop-Weed Model Loaded Successfully!")
else:
    st.sidebar.warning(" Using baseline weights. 'best.pt' file not detected in project folder.")

# ==============================================================================
# 3. FILE UPLOAD & PRE-PROCESSING MANAGEMENT
# ==============================================================================
uploaded_file = st.file_uploader(
    label="Upload Field Vegetation Image Frame", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert file stream directly into an OpenCV matrix image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Standardize image array to RGB color space mapping for Streamlit visualization
    display_rgb_image = cv2.cvtColor(raw_bgr_image, cv2.COLOR_BGR2RGB)
    
    # Render layout columns to contrast Input vs Mask Outputs cleanly
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Input Frame")
        st.image(display_rgb_image, use_container_width=True)
        
    with col2:
        st.subheader("Prediction Target Output")
        
        # Trigger explicit button verification pattern to optimize compute cost
        if st.button("Execute Targeted Inference Mapping", type="primary"):
            with st.spinner("Parsing neural features and compiling bounding segments..."):
                
                # Run the model inference pipeline execution pass
                inference_results = model(display_rgb_image, imgsz=512)
                
                # Extract the first result object matrix array loop frame 
                result_item = inference_results[0]
                
                # Render the built-in bounding box overlays dynamically
                annotated_rgb_output = result_item.plot()
                
                # Render processed output image frame visually inside column 2
                st.image(annotated_rgb_output, use_container_width=True)
                
                # ==============================================================
                # 4. METRICS METADATA COMPILATION
                # ==============================================================
                # Count instances of detections based on classification index integers
                detected_classes = result_item.boxes.cls.cpu().numpy().astype(int) if len(result_item.boxes) > 0 else []
                
                crop_count = np.sum(detected_classes == 0)
                weed_count = np.sum(detected_classes == 1)
                
                st.markdown("### Target Density Statistics")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(label="Detected Crop Targets", value=int(crop_count))
                with metric_col2:
                    st.metric(label="Detected Weed Targets", value=int(weed_count))
                
                # System Status Recommendations based on vegetation densities
                if weed_count > 0:
                    st.error(f" Action Required: {weed_count} weeds detected. Activating localized sprayer arrays.")
                else:
                    st.success(" Standby: No weeds detected in this sector view frame.")