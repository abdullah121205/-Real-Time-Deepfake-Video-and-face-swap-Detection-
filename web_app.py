"""
Real-Time Deepfake and Face-Swap Detection - Advanced Forensic Web Dashboard.
Enterprise-grade interface for video verification, deep forensic spectral analysis,
live camera streaming, batch queue inspection, and latency profiling.
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.deepfake_detector import DeepfakeDetector
from src.face_detector import FaceDetector
from src.face_swap_detector import FaceSwapDetector
from src.utils import TemporalSmoother, crop_face_with_margin, draw_detection_hud
from src.video_processor import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Deepfake & Face-Swap Cyber Forensics Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk & Glassmorphism Forensics Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');

    .reportview-container, .main {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 23, 42, 0.95) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    .card-glass {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(6px);
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-authentic {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-suspicious {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-deepfake {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-faceswap {
        background: rgba(217, 70, 239, 0.18);
        color: #e879f9;
        border: 1px solid rgba(217, 70, 239, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Forensic Spectral Visualization Helpers
def generate_fft_spectrum(crop_bgr: np.ndarray) -> np.ndarray:
    """Computes and colorizes the 2D Fourier power spectrum heatmap."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128)).astype(np.float32)
    dft = np.fft.fft2(resized)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log(np.abs(dft_shift) + 1.0)
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def generate_edge_heatmap(crop_bgr: np.ndarray) -> np.ndarray:
    """Extracts Sobel gradient boundary seams for face-swap boundary detection."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def generate_chroma_disparity_map(crop_bgr: np.ndarray) -> np.ndarray:
    """Maps chromaticity variance in CIELAB color space to highlight skin tone mismatch."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    chroma = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)
    norm = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Initializing Deepfake Forensic Engine...")
def load_models(model_name: str, face_backend: str, deepfake_weight: float, swap_weight: float):
    """Caches neural model instances for rapid session reuse."""
    weights_path = None
    for candidate in [f"models/{model_name}_trained.pth", f"models/{model_name}_default.pth", f"models/{model_name}.pth"]:
        if Path(candidate).is_file():
            weights_path = candidate
            break

    face_detector = FaceDetector(backend=face_backend)
    deepfake_detector = DeepfakeDetector(model_name=model_name, weights_path=weights_path)
    face_swap_detector = FaceSwapDetector()
    smoother = TemporalSmoother(window_size=12, alpha=0.35)

    processor = VideoProcessor(
        face_detector=face_detector,
        deepfake_detector=deepfake_detector,
        face_swap_detector=face_swap_detector,
        temporal_smoother=smoother,
        frame_stride=1,
        deepfake_weight=deepfake_weight,
        swap_weight=swap_weight
    )
    return processor, deepfake_detector, face_swap_detector


def render_sidebar():
    st.sidebar.markdown("### 🛡️ Cyber Forensics Controls")

    mode = st.sidebar.selectbox(
        "Select Operation Mode:",
        [
            "🎬 Video Deep Scan",
            "🔬 Forensic Spectral Lab",
            "🎥 Live Camera Surveillance",
            "📁 Batch Media Scanner",
            "⚡ Hardware & Latency Profiler"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🧠 Model Configurations")
    model_choice = st.sidebar.selectbox(
        "Deepfake Neural Core:",
        ["meso4", "meso_inception", "mobilenet"],
        index=0,
        help="Meso4: High FPS, mesoscopic features. MesoInception4: Dilated multi-scale inception. MobileNet: Deep architecture."
    )

    backend_choice = st.sidebar.selectbox(
        "Face Detector Backend:",
        ["yunet", "haar"],
        index=0,
        help="YuNet: High-accuracy OpenCV DNN ONNX model. Haar: Classic fast CPU cascade."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚖️ Sensitivity & Weights")
    threshold = st.sidebar.slider("Decision Threshold (Risk Cutoff)", 0.30, 0.80, 0.50, 0.05,
                                   help="Scores above this threshold are classified as manipulated.")
    df_weight = st.sidebar.slider("Neural Classifier Weight", 0.0, 1.0, 0.65, 0.05)
    swap_weight = st.sidebar.slider("Face-Swap Forensics Weight", 0.0, 1.0, 0.35, 0.05)
    stride = st.sidebar.slider("Frame Stride", 1, 4, 1, 1, help="Process every Nth frame for acceleration")

    device_str = "CUDA (NVIDIA GPU Accelerated)" if torch.cuda.is_available() else "CPU (Standard Mode)"
    st.sidebar.info(f"Engine: **{device_str}**")

    return mode, model_choice, backend_choice, threshold, df_weight, swap_weight, stride


# Mode 1: Video Deep Scan
def run_video_deep_scan(video_path: str, processor: VideoProcessor, threshold: float, stride: int):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processor.frame_stride = stride

    st.markdown("### 🎥 Real-Time Deepfake Video Telemetry")

    col_view, col_stats = st.columns([2, 1])

    with col_view:
        video_placeholder = st.empty()
    with col_stats:
        kpi_verdict = st.empty()
        kpi_score = st.empty()
        kpi_faces = st.empty()
        kpi_fps = st.empty()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    frame_idx = 0
    df_scores, swap_scores, combined_scores, timestamps = [], [], [], []
    detailed_telemetry_log = []

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out_path = temp_out.name
    temp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    t_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated_frame, telemetry = processor.process_frame(frame, show_details=True)
        writer.write(annotated_frame)

        detections = telemetry.get("detections", [])
        if detections:
            top_det = max(detections, key=lambda d: d["fake_score"])
            c_score = top_det["fake_score"]
            d_score = top_det["deepfake_score"]
            s_score = top_det["swap_score"]
            v_label = top_det["label"]
        else:
            c_score, d_score, s_score = 0.0, 0.0, 0.0
            v_label = "AUTHENTIC"

        combined_scores.append(c_score)
        df_scores.append(d_score)
        swap_scores.append(s_score)
        t_sec = frame_idx / fps
        timestamps.append(t_sec)

        detailed_telemetry_log.append({
            "frame": frame_idx,
            "timestamp_sec": round(t_sec, 2),
            "combined_score": round(c_score, 4),
            "deepfake_score": round(d_score, 4),
            "faceswap_score": round(s_score, 4),
            "verdict": v_label,
            "num_faces": len(detections)
        })

        if frame_idx % 2 == 0 or frame_idx == total_frames:
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if c_score >= threshold:
                badge_class = "badge-faceswap" if s_score > 0.60 else "badge-deepfake"
            elif c_score >= (threshold * 0.75):
                badge_class = "badge-suspicious"
            else:
                badge_class = "badge-authentic"

            kpi_verdict.markdown(f"**Current Frame:** <span class='status-badge {badge_class}'>{v_label}</span>", unsafe_allow_html=True)
            kpi_score.metric("Manipulation Score", f"{c_score * 100:.1f}%")
            kpi_faces.metric("Detected Faces", telemetry.get("num_faces", 0))
            kpi_fps.metric("Processing Throughput", f"{telemetry.get('fps', 0):.1f} FPS")

            progress_bar.progress(min(1.0, frame_idx / total_frames))
            status_text.text(f"Scanning frame {frame_idx}/{total_frames} ({(frame_idx / total_frames) * 100:.1f}%)")

    cap.release()
    writer.release()
    total_elapsed = time.perf_counter() - t_start

    status_text.success(f"Scan complete: {frame_idx} frames processed in {total_elapsed:.2f}s ({frame_idx / total_elapsed:.1f} FPS).")

    # Interactive Waveform Chart
    st.markdown("---")
    st.markdown("### 📈 Multi-Signal Temporal Waveform")

    chart_df = pd.DataFrame({
        "Timestamp (s)": timestamps,
        "Combined Risk Score": combined_scores,
        "Neural Deepfake Core": df_scores,
        "Face-Swap Spectral Anomaly": swap_scores
    }).set_index("Timestamp (s)")

    st.line_chart(chart_df)

    # Forensic Summary Assessment
    st.markdown("### 📑 Digital Forensics Audit Certificate")

    mean_score = float(np.mean(combined_scores)) if combined_scores else 0.0
    peak_score = float(np.max(combined_scores)) if combined_scores else 0.0
    mean_swap = float(np.mean(swap_scores)) if swap_scores else 0.0

    if peak_score >= threshold or mean_score >= (threshold * 0.9):
        final_verdict = "HIGH-RISK FORGERY DETECTED"
        badge_style = "badge-deepfake"
    elif peak_score >= (threshold * 0.75):
        final_verdict = "SUSPICIOUS / ELEVATED ANOMALIES"
        badge_style = "badge-suspicious"
    else:
        final_verdict = "VERIFIED AUTHENTIC MEDIA"
        badge_style = "badge-authentic"

    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
    with rep_col1:
        st.markdown(f"<span class='status-badge {badge_style}'>{final_verdict}</span>", unsafe_allow_html=True)
    with rep_col2:
        st.metric("Mean Manipulation Probability", f"{mean_score * 100:.1f}%")
    with rep_col3:
        st.metric("Peak Manipulation Spike", f"{peak_score * 100:.1f}%")
    with rep_col4:
        st.metric("Face-Swap Boundary Distortion", f"{mean_swap * 100:.1f}%")

    # Download Buttons (Video, CSV, JSON Forensic Log)
    st.markdown("---")
    st.markdown("#### 📥 Export Forensic Artifacts")
    d_col1, d_col2, d_col3 = st.columns(3)

    with open(temp_out_path, "rb") as f:
        video_bytes = f.read()

    with d_col1:
        st.download_button(
            "📥 Download Annotated Video (.mp4)",
            data=video_bytes,
            file_name="forensic_annotated_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # CSV Telemetry Export
    csv_buffer = io.StringIO()
    pd.DataFrame(detailed_telemetry_log).to_csv(csv_buffer, index=False)
    with d_col2:
        st.download_button(
            "📊 Download Frame Telemetry (CSV)",
            data=csv_buffer.getvalue(),
            file_name="telemetry_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Forensic JSON Audit Log
    json_report = {
        "report_id": f"DF-{int(time.time())}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "verdict": final_verdict,
            "mean_score": round(mean_score, 4),
            "peak_score": round(peak_score, 4),
            "total_frames_analyzed": frame_idx,
            "processing_duration_seconds": round(total_elapsed, 2),
            "effective_fps": round(frame_idx / total_elapsed, 2)
        },
        "frames": detailed_telemetry_log
    }
    with d_col3:
        st.download_button(
            "📜 Download Forensic Audit (JSON)",
            data=json.dumps(json_report, indent=2),
            file_name="forensic_audit_log.json",
            mime="application/json",
            use_container_width=True
        )


# Mode 2: Deep Forensic Spectral Lab
def run_forensic_spectral_lab(image_bgr: np.ndarray, processor: VideoProcessor):
    st.markdown("### 🔬 Multi-Spectral & Boundary Forensic Analysis")

    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.markdown("**Original Ingestion Image**")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    annotated, telemetry = processor.process_frame(image_bgr, show_details=True)

    with view_col2:
        st.markdown("**Forensic Telemetry HUD**")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    detections = telemetry.get("detections", [])
    if not detections:
        st.warning("No face detected in this frame. Check detector backend in sidebar.")
        return

    st.markdown("---")
    st.markdown(f"#### 🧬 Spectral Decomposition ({len(detections)} Face Targets)")

    for i, det in enumerate(detections, 1):
        raw_box = det.get("raw_bbox", det["bbox"])
        crop, _ = crop_face_with_margin(image_bgr, raw_box, margin=0.25)
        if crop is None or crop.size == 0:
            continue

        with st.expander(f"Face Target #{i} — Verdict: {det['label']} (Risk: {det['fake_score'] * 100:.1f}%)", expanded=True):
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)

            with fcol1:
                st.markdown("**Facial Crop ROI**")
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

            with fcol2:
                st.markdown("**2D-FFT Power Spectrum**")
                fft_vis = generate_fft_spectrum(crop)
                st.image(fft_vis, use_container_width=True, caption="High-Freq Artifacts")

            with fcol3:
                st.markdown("**Boundary Sobel Gradients**")
                edge_vis = generate_edge_heatmap(crop)
                st.image(edge_vis, use_container_width=True, caption="Seam Discontinuities")

            with fcol4:
                st.markdown("**CIELAB Chroma Variance**")
                chroma_vis = generate_chroma_disparity_map(crop)
                st.image(chroma_vis, use_container_width=True, caption="Skin Tone Disparities")

            st.markdown("##### 📊 Signal Forensic Decomposition")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Neural Deepfake Core", f"{det['deepfake_score'] * 100:.1f}%")
            m_col2.metric("Face-Swap Boundary Ratio", f"{det.get('boundary_anomaly', 0.0):.3f}")
            m_col3.metric("FFT Frequency Disparity", f"{det.get('freq_score', 0.0):.3f}")
            m_col4.metric("Composite Risk Score", f"{det['fake_score'] * 100:.1f}%")

            # Mini Bar chart comparison
            sig_df = pd.DataFrame({
                "Forensic Signal": ["Neural Core", "Boundary Seams", "Spectral FFT", "Composite Risk"],
                "Score (0.0 to 1.0)": [
                    det["deepfake_score"],
                    det.get("boundary_anomaly", 0.0),
                    det.get("freq_score", 0.0),
                    det["fake_score"]
                ]
            }).set_index("Forensic Signal")
            st.bar_chart(sig_df)


# Mode 3: Live Camera Surveillance Stream
def run_live_surveillance(processor: VideoProcessor):
    st.markdown("### 🎥 Live Camera Surveillance Stream")
    st.write("Stream directly from your camera with real-time HUD annotations and snapshot logging:")

    cam_mode = st.radio("Select Capture Method:", ["🔴 Interactive Browser Snapshot", "⚡ Direct Local OpenCV Stream (Desktop)"], horizontal=True)

    if cam_mode == "🔴 Interactive Browser Snapshot":
        camera_img = st.camera_input("Acquire Live Frame")
        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            run_forensic_spectral_lab(frame_bgr, processor)

    else:
        st.info("Live desktop streaming will open your local device camera feed (Index 0). Press Stop Stream to pause.")
        stream_placeholder = st.empty()
        run_stream = st.checkbox("Start Live Video Stream", value=False)

        if run_stream:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access camera (Index 0). Please ensure camera permissions are enabled.")
                return

            stop_btn = st.button("⏹️ Stop Stream")
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, telemetry = processor.process_frame(frame, show_details=True)
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stream_placeholder.image(frame_rgb, use_container_width=True)
                time.sleep(0.01)

            cap.release()
            stream_placeholder.empty()


# Mode 4: Batch Media Scanner
def run_batch_scanner(processor: VideoProcessor):
    st.markdown("### 📁 Batch Media Forensic Scanner")
    st.write("Inspect multiple video clips or image files simultaneously in queue mode:")

    uploaded_files = st.file_uploader(
        "Upload multiple files (Images / Videos):",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** files for verification.")

        if st.button("🚀 Process Batch Queue", type="primary"):
            results_summary = []
            batch_bar = st.progress(0.0)

            for idx, file in enumerate(uploaded_files):
                filename = file.name
                ext = Path(filename).suffix.lower()

                if ext in [".jpg", ".jpeg", ".png"]:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    _, tel = processor.process_frame(img)
                    dets = tel.get("detections", [])
                    max_score = max([d["fake_score"] for d in dets]) if dets else 0.0
                    verdict = max([d["label"] for d in dets]) if dets else "AUTHENTIC"
                else:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(file.read())
                    tfile.close()

                    cap = cv2.VideoCapture(tfile.name)
                    v_scores = []
                    while cap.isOpened():
                        ret, f = cap.read()
                        if not ret:
                            break
                        _, tel = processor.process_frame(f)
                        dets = tel.get("detections", [])
                        if dets:
                            v_scores.append(max([d["fake_score"] for d in dets]))
                    cap.release()
                    max_score = max(v_scores) if v_scores else 0.0
                    verdict = "DEEPFAKE" if max_score >= 0.50 else "AUTHENTIC"

                results_summary.append({
                    "File Name": filename,
                    "Format": ext.upper().replace(".", ""),
                    "Peak Risk Score": f"{max_score * 100:.1f}%",
                    "Status Verdict": verdict
                })
                batch_bar.progress((idx + 1) / len(uploaded_files))

            st.success("Batch scan complete!")
            res_df = pd.DataFrame(results_summary)
            st.dataframe(res_df, use_container_width=True)

            csv_data = res_df.to_csv(index=False)
            st.download_button(
                "📥 Download Batch Report (CSV)",
                data=csv_data,
                file_name="batch_forensic_report.csv",
                mime="text/csv"
            )


# Mode 5: Hardware & Latency Profiler
def run_latency_profiler(processor: VideoProcessor):
    st.markdown("### ⚡ Hardware & Real-Time Latency Profiler")
    st.write("Benchmark system inference latency across 720p (1280x720) simulated frames:")

    num_frames = st.slider("Test Frame Count", 30, 300, 100, 20)

    if st.button("▶️ Execute Profiling Suite", type="primary"):
        sim_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        cv2.circle(sim_frame, (640, 360), 120, (180, 200, 230), -1)
        cv2.circle(sim_frame, (590, 320), 15, (50, 50, 50), -1)
        cv2.circle(sim_frame, (690, 320), 15, (50, 50, 50), -1)
        cv2.ellipse(sim_frame, (640, 420), (50, 20), 0, 0, 180, (40, 40, 150), -1)

        latencies = []
        bar = st.progress(0.0)
        t_start = time.perf_counter()

        for i in range(num_frames):
            t0 = time.perf_counter()
            processor.process_frame(sim_frame, show_details=True)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            bar.progress((i + 1) / num_frames)

        total_time = time.perf_counter() - t_start
        latencies_arr = np.array(latencies)

        avg_lat = np.mean(latencies_arr)
        p50_lat = np.percentile(latencies_arr, 50)
        p90_lat = np.percentile(latencies_arr, 90)
        p95_lat = np.percentile(latencies_arr, 95)
        p99_lat = np.percentile(latencies_arr, 99)
        eff_fps = num_frames / total_time

        st.success("Profiling complete!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Effective Throughput", f"{eff_fps:.1f} FPS")
        c2.metric("Median (p50) Latency", f"{p50_lat:.2f} ms")
        c3.metric("95th Pct Latency", f"{p95_lat:.2f} ms")
        c4.metric("99th Pct Latency", f"{p99_lat:.2f} ms")

        st.markdown("#### 📉 Frame Latency Distribution (ms)")
        st.line_chart(pd.DataFrame({"Latency (ms)": latencies}))

        if eff_fps >= 25.0:
            st.info("System Assessment: **READY FOR LIVE REAL-TIME STREAMING (>= 25 FPS)**.")
        else:
            st.warning("System Assessment: Use **Frame Stride = 2** in sidebar for 60Hz/high-framerate cameras.")


# Main Application Entry Point
def main():
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>🛡️ Deepfake & Face-Swap Cyber Forensics Lab</div>
        <div class='hero-subtitle'>
            Real-time neural and multi-spectral computer vision engine for authenticating digital media, 
            detecting GAN/diffusion synthesis, and identifying facial boundary manipulation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode, model_name, face_backend, threshold, df_weight, swap_weight, stride = render_sidebar()

    processor, deepfake_detector, face_swap_detector = load_models(
        model_name=model_name,
        face_backend=face_backend,
        deepfake_weight=df_weight,
        swap_weight=swap_weight
    )

    if mode == "🎬 Video Deep Scan":
        st.markdown("#### Video File Verification")
        vcol1, vcol2 = st.columns([3, 1])
        with vcol1:
            uploaded_file = st.file_uploader("Upload video file (MP4, AVI, MOV):", type=["mp4", "avi", "mov"])
        with vcol2:
            st.write("&nbsp;")
            use_demo = st.button("📁 Load Demo Video (data/demo_sample.mp4)")

        target_path = None
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            target_path = tfile.name
        elif use_demo or Path("data/demo_sample.mp4").is_file():
            target_path = "data/demo_sample.mp4"
            if not uploaded_file:
                st.info("Pre-loaded demo video: `data/demo_sample.mp4` (Authentic face transitioning into Face-Swap manipulation).")

        if target_path:
            if st.button("🚀 Start Deep Forensic Scan", type="primary"):
                run_video_deep_scan(target_path, processor, threshold, stride)

    elif mode == "🔬 Forensic Spectral Lab":
        st.markdown("#### High-Resolution Face & Spectrum Analysis")
        icol1, icol2 = st.columns([3, 1])
        with icol1:
            uploaded_img = st.file_uploader("Upload target image:", type=["jpg", "jpeg", "png"])
        with icol2:
            sample_choice = st.radio("Demo Sample:", ["None", "Authentic Face", "Face-Swap Face"])

        img_bgr = None
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif sample_choice == "Authentic Face" and Path("data/sample_real.jpg").is_file():
            img_bgr = cv2.imread("data/sample_real.jpg")
        elif sample_choice == "Face-Swap Face" and Path("data/sample_fake.jpg").is_file():
            img_bgr = cv2.imread("data/sample_fake.jpg")

        if img_bgr is not None:
            run_forensic_spectral_lab(img_bgr, processor)

    elif mode == "🎥 Live Camera Surveillance":
        run_live_surveillance(processor)

    elif mode == "📁 Batch Media Scanner":
        run_batch_scanner(processor)

    elif mode == "⚡ Hardware & Latency Profiler":
        run_latency_profiler(processor)


if __name__ == "__main__":
    main()
