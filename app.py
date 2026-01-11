"""
Real-Time Object Detection with YOLOv8
A Streamlit web app that detects objects in images and video using YOLOv8.
Features:
- Upload single images, batches, videos or webcam feed for detection
- Adjust model size, confidence threshold, and preprocessing settings
- View results with bounding boxes, confidence heatmaps, and charts
- Export detections as CSV, JSON, or annotated images       
"""

import io
import json
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# Load Model (cached so it only loads once per model size)
# ============================================================
@st.cache_resource
def load_model(model_name):
    return YOLO(f"{model_name}.pt")


# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.title("⚙️ Settings")

model_size = st.sidebar.selectbox(
    "Model Size",
    ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    index=0,
    help="Nano (fastest) → Small → Medium → Large → XLarge (most accurate)"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1, max_value=1.0, value=0.5, step=0.05,
    help="Only show detections above this confidence"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1, max_value=1.0, value=0.45, step=0.05,
    help="Non-Maximum Suppression threshold"
)

model = load_model(model_size)
all_classes = list(model.names.values())

selected_classes = st.sidebar.multiselect(
    "Filter Classes (leave empty for all)",
    options=all_classes,
    default=[],
    help="Only show these object types"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Image Preprocessing**")

brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1,
                               help="Adjust brightness before detection")
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1,
                             help="Adjust contrast before detection")
sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1,
                              help="Adjust sharpness before detection")

st.sidebar.markdown("---")
show_heatmap = st.sidebar.checkbox("Show Detection Heatmap", value=False,
                                   help="Overlay a confidence-weighted heatmap instead of boxes")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Architecture: **{model_size.upper()}**")
st.sidebar.markdown(f"Classes: **{len(all_classes)}** (COCO dataset)")


# ============================================================
# Helper Functions
# ============================================================

def apply_preprocessing(image: Image.Image) -> Image.Image:
    """Apply brightness / contrast / sharpness adjustments."""
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def generate_heatmap(image_rgb: np.ndarray, boxes) -> np.ndarray:
    """Generate a confidence-weighted detection heatmap blended onto the image."""
    h, w = image_rgb.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        heatmap[y1:y2, x1:x2] += float(box.conf)

    heatmap = gaussian_filter(heatmap, sigma=20)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)


def build_detections_df(boxes, model_names, filter_classes=None) -> pd.DataFrame:
    """Build a DataFrame of bounding box detections."""
    rows = []
    for box in boxes:
        cls_name = model_names[int(box.cls)]
        if filter_classes and cls_name not in filter_classes:
            continue
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]
        rows.append({
            "Class": cls_name,
            "Confidence": round(float(box.conf), 4),
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
            "Width": round(x2 - x1, 1),
            "Height": round(y2 - y1, 1),
        })
    return pd.DataFrame(rows)


def show_charts(df: pd.DataFrame, suffix: str = ""):
    """Bar chart of detections per class + confidence histogram."""
    if df.empty:
        return
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        counts = df["Class"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.bar(counts, x="Class", y="Count", title="Detections per Class",
                     color="Count", color_continuous_scale="viridis")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{suffix}")

    with chart_col2:
        fig = px.histogram(df, x="Confidence", nbins=20,
                           title="Confidence Score Distribution",
                           color_discrete_sequence=["#636EFA"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{suffix}")


def show_export_buttons(df: pd.DataFrame, annotated_rgb: np.ndarray, suffix: str = ""):
    """CSV, JSON, and annotated-image download buttons."""
    st.markdown("#### Export Results")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.download_button(
            "📄 Download CSV",
            data=df.to_csv(index=False),
            file_name="detections.csv",
            mime="text/csv",
            key=f"download_csv_{suffix}"
        )
    with exp_col2:
        st.download_button(
            "📋 Download JSON",
            data=df.to_json(orient="records", indent=2),
            file_name="detections.json",
            mime="application/json",
            key=f"download_json_{suffix}"
        )
    with exp_col3:
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            "📥 Download Image",
            data=buf.getvalue(),
            file_name="detected.png",
            mime="image/png",
            key=f"download_image_{suffix}"
        )


def run_detection_and_display(image: Image.Image, label: str = "") -> pd.DataFrame:
    """
    Preprocess → detect → display original + annotated/heatmap side by side.
    Returns a DataFrame of detections (empty if none found).
    """
    processed = apply_preprocessing(image)
    img_array = np.array(processed)

    with st.spinner("Detecting objects..."):
        results = model(img_array, conf=confidence, iou=iou_threshold)

    boxes = results[0].boxes
    annotated_bgr = results[0].plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    col1, col2 = st.columns(2)
    col1.image(image, caption=f"Original{' — ' + label if label else ''}", use_container_width=True)

    if show_heatmap and len(boxes) > 0:
        heatmap_img = generate_heatmap(np.array(image.convert("RGB")), boxes)
        col2.image(heatmap_img, caption="Detection Heatmap", use_container_width=True)
    else:
        col2.image(annotated_rgb, caption="Detections", use_container_width=True)

    df = build_detections_df(boxes, model.names, selected_classes)

    if not df.empty:
        st.markdown("### Detection Results")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Objects Detected", len(df))
        mcol2.metric("Unique Classes", df["Class"].nunique())
        mcol3.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")
        show_charts(df, suffix=f"main_{label}")
        st.dataframe(df, use_container_width=True)
        st.markdown("---")
        show_export_buttons(df, annotated_rgb, suffix=f"main_{label}")
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

    return df


# ============================================================
# Main Content : Tabbed Interface
# ============================================================
st.title("🔍 Real-Time Object Detection")
st.markdown("Detect objects in images or video using **YOLOv8**.")

tab_image, tab_batch, tab_video, tab_webcam = st.tabs(
    ["📷 Single Image", "🖼️ Batch Images", "🎬 Video", "📹 Webcam"]
)

# ──────────────────────────────────────────────────────
# TAB 1 : Single Image
# ──────────────────────────────────────────────────────
with tab_image:
    use_sample = st.checkbox("Use sample image (YOLO bus demo)")

    if use_sample:
        sample_url = "https://ultralytics.com/images/bus.jpg"
        st.info("Using sample image from YOLO documentation")
        with st.spinner("Detecting objects..."):
            results = model(sample_url, conf=confidence, iou=iou_threshold)

        boxes = results[0].boxes
        orig_rgb = results[0].orig_img[:, :, ::-1]
        annotated_rgb = results[0].plot()[:, :, ::-1]

        col1, col2 = st.columns(2)
        col1.image(orig_rgb, caption="Original Image", use_container_width=True)

        if show_heatmap and len(boxes) > 0:
            col2.image(generate_heatmap(orig_rgb.copy(), boxes),
                       caption="Detection Heatmap", use_container_width=True)
        else:
            col2.image(annotated_rgb, caption="Detections", use_container_width=True)

        df = build_detections_df(boxes, model.names, selected_classes)
        if not df.empty:
            st.markdown("### Detection Results")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Objects Detected", len(df))
            mcol2.metric("Unique Classes", df["Class"].nunique())
            mcol3.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")
            show_charts(df, suffix="sample")
            st.dataframe(df, use_container_width=True)
            st.markdown("---")
            show_export_buttons(df, annotated_rgb, suffix="sample")

    else:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supports JPG, JPEG, PNG, WEBP"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            run_detection_and_display(image)
        else:
            st.markdown("""
### How to Use
1. **Upload** an image or tick the sample checkbox above
2. **Adjust** settings in the sidebar (model size, confidence, preprocessing)
3. **View** detections with bounding boxes, charts, and optional heatmap
4. **Export** results as CSV / JSON / annotated image

### Supported Objects
YOLOv8 detects **80 object classes** from the COCO dataset including:
person, car, bus, truck, bicycle, dog, cat, chair, laptop, and many more.
            """)

# ──────────────────────────────────────────────────────
# TAB 2 : Batch Images
# ──────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Upload multiple images and get detection results for all of them.")

    batch_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:
        all_dfs = []
        for idx, f in enumerate(batch_files):
            st.markdown(f"---\n#### Image {idx + 1}: {f.name}")
            image = Image.open(f).convert("RGB")
            df = run_detection_and_display(image, label=f.name)
            if df is not None and not df.empty:
                df.insert(0, "Image", f.name)
                all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            st.markdown("---")
            st.markdown("## Aggregate Results — All Images")
            show_charts(combined_df, suffix="batch_aggregate")
            st.download_button(
                "📄 Download All Results (CSV)",
                data=combined_df.to_csv(index=False),
                file_name="batch_detections.csv",
                mime="text/csv"
            )

# ──────────────────────────────────────────────────────
# TAB 3 : Video (detection + optional tracking)
# ──────────────────────────────────────────────────────
with tab_video:
    st.markdown("Upload a video for frame-by-frame object detection with optional tracking.")

    use_tracking = st.checkbox(
        "Enable Object Tracking (assigns persistent IDs across frames)", value=True
    )
    max_frames = st.slider(
        "Max frames to process", 10, 300, 100, 10,
        help="Limit frames to keep processing time reasonable"
    )

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(video_file.read())
            tmp_in_path = tmp_in.name

        cap = cv2.VideoCapture(tmp_in_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(f"Video: {total_frames} frames | {fps:.1f} FPS | {width}×{height}")

        frames_to_process = min(max_frames, total_frames)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            tmp_out_path = tmp_out.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0, text="Processing video...")
        all_video_rows = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if use_tracking:
                results = model.track(
                    frame, conf=confidence, iou=iou_threshold,
                    persist=True, verbose=False
                )
            else:
                results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)

            out.write(results[0].plot())

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls)]
                if selected_classes and cls_name not in selected_classes:
                    continue
                row = {
                    "Frame": frame_idx,
                    "Class": cls_name,
                    "Confidence": round(float(box.conf), 4),
                }
                if use_tracking and box.id is not None:
                    row["Track ID"] = int(box.id)
                all_video_rows.append(row)

            frame_idx += 1
            progress_bar.progress(
                frame_idx / frames_to_process,
                text=f"Processing frame {frame_idx}/{frames_to_process}"
            )

        cap.release()
        out.release()
        progress_bar.empty()
        os.unlink(tmp_in_path)

        st.success(f"Processed {frame_idx} frames!")

        with open(tmp_out_path, "rb") as f:
            st.download_button(
                "🎬 Download Annotated Video",
                data=f.read(),
                file_name="detected_video.mp4",
                mime="video/mp4"
            )
        os.unlink(tmp_out_path)

        if all_video_rows:
            video_df = pd.DataFrame(all_video_rows)
            st.markdown("### Video Detection Summary")

            vcol1, vcol2, vcol3 = st.columns(3)
            vcol1.metric("Total Detections", len(video_df))
            vcol2.metric("Unique Classes", video_df["Class"].nunique())
            if use_tracking and "Track ID" in video_df.columns:
                vcol3.metric("Unique Tracked Objects", video_df["Track ID"].nunique())
            else:
                vcol3.metric("Frames Processed", frame_idx)

            show_charts(video_df, suffix="video")

            # Detections over time line chart
            by_frame = video_df.groupby("Frame").size().reset_index(name="Detections")
            fig_time = px.line(by_frame, x="Frame", y="Detections",
                               title="Detections per Frame Over Time")
            st.plotly_chart(fig_time, use_container_width=True)

            st.download_button(
                "📄 Download Video Detections (CSV)",
                data=video_df.to_csv(index=False),
                file_name="video_detections.csv",
                mime="text/csv"
            )

# ──────────────────────────────────────────────────────
# TAB 4 : Webcam
# ──────────────────────────────────────────────────────
with tab_webcam:
    st.markdown("Capture a photo from your webcam and detect objects instantly.")
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        run_detection_and_display(image, label="Webcam Capture")


# Footer
st.markdown("---")
st.markdown(
    "Built with [YOLOv8](https://github.com/ultralytics/ultralytics) + "
    "[Streamlit](https://streamlit.io) | "
    "[GitHub](https://github.com/GamithaManawadu)"
)
