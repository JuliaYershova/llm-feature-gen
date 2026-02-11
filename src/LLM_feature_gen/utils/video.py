# src/LLM_feature_gen/utils/video.py
import os
import time
import requests
import ffmpeg
import cv2
import base64
import io
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


def _get_frame_signature(image: np.ndarray) -> np.ndarray:
    """
    Creates a 'fingerprint' of the image combining color (HSV) and structure.
    Helps group similar shots (e.g., zoom-in vs zoom-out of the same building).
    """
    # 1. Color profile (HSV is robust to lighting changes)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # 2. Structural layout (tiny thumbnail)
    small = cv2.resize(image, (16, 16))
    small_flat = small.flatten().astype(np.float32) / 255.0

    # Combine them (giving more weight to color histogram)
    return np.concatenate([hist.flatten() * 5, small_flat])


def extract_key_frames(video_path: str, min_frames: int = 5, max_frames: int = 10, sharpness_threshold: float = 40.0) -> List[str]:
    """
    Selects diverse keyframes using K-Means clustering.
    Instead of looking for motion, it groups similar scenes and picks the sharpest image from each group.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_limit = int(duration / 5)
    frame_limit = max(min_frames, min(frame_limit, max_frames))

    # Step 1: Gather candidates (~2 frames per second to be efficient)
    sample_rate = max(1, int(fps / 2))
    candidates = []
    frame_idx = 0

    while True:
        is_read, frame = cap.read()
        if not is_read:
            break

        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        # Skip blurry frames immediately
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < sharpness_threshold:
            continue

        candidates.append({
            "frame": frame,
            "timestamp": frame_idx / fps,
            "sharpness": sharpness,
            "signature": _get_frame_signature(frame)
        })

    cap.release()

    if not candidates:
        return []

    # Intelligent Selection
    if len(candidates) <= frame_limit:
        # Not enough candidates? Take them all.
        final_candidates = candidates
    else:
        # Group frames by visual similarity
        data = np.array([c["signature"] for c in candidates], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Run K-Means
        _, labels, _ = cv2.kmeans(data, frame_limit, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        selected_indices = []
        for i in range(frame_limit):
            # Find all frames belonging to this cluster
            cluster_indices = [idx for idx, label in enumerate(labels) if label == i]

            if not cluster_indices:
                continue

            # Pick the sharpest frame from this cluster
            best_in_cluster = max(cluster_indices, key=lambda idx: candidates[idx]["sharpness"])
            selected_indices.append(best_in_cluster)

        final_candidates = [candidates[i] for i in selected_indices]

    # Sort by time and prepare output
    final_candidates.sort(key=lambda x: x["timestamp"])

    b64_list = []
    for item in final_candidates:
        frame = item["frame"]

        # Burn-in timestamp for the LLM
        seconds = int(item["timestamp"])
        time_str = f"{seconds // 60:02d}:{seconds % 60:02d}"

        # Draw text (black border + white text for readability)
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert to base64
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

    return b64_list


def extract_audio_track(file_path: str) -> Optional[str]:
    """
    Extracts the audio track from a video file and saves it as a temporary WAV file.
    Uses FFmpeg to convert the stream to mono, 16kHz PCM (standard for Whisper/STT).

    Returns:
        The path to the generated temporary WAV file, or None if extraction fails.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Create a unique temporary filename
    temp_audio_path = f"temp_audio_{base_name}_{int(time.time())}.wav"

    try:
        # ffmpeg-python configuration:
        # ac=1 (mono), ar=16000 (16kHz), acodec='pcm_s16le' (linear PCM)
        ffmpeg.input(file_path).output(
            temp_audio_path,
            acodec='pcm_s16le',
            ac=1,
            ar='16000'
        ).run(quiet=True, overwrite_output=True)

        if os.path.exists(temp_audio_path):
            return temp_audio_path
        return None
    except Exception as e:
        print(f"Error extracting audio from {file_path}: {e}")
        return None