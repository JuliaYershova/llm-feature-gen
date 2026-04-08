# src/llm_feature_gen/utils/video.py
import os
import time
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


def downsample_batch(b64_list: List[str], target_count: int = 15) -> List[str]:
    """
    Takes a large list of base64 images (e.g. from multiple videos) and
    selects the most diverse set using K-Means clustering.
    """
    if len(b64_list) <= target_count:
        return b64_list

    candidates = []

    # 1. Decode base64 back to CV2 for analysis (fast enough for <100 images)
    for idx, b64 in enumerate(b64_list):
        try:
            img_data = base64.b64decode(b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None: continue

            candidates.append({
                "original_b64": b64,
                "signature": _get_frame_signature(img),
                "original_idx": idx
            })
        except Exception:
            continue

    if not candidates:
        return b64_list[:target_count]

    # 2. K-Means Clustering on the signatures
    data = np.array([c["signature"] for c in candidates], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Handle edge case where we have fewer candidates than clusters (shouldn't happen due to if check above)
    K = min(target_count, len(candidates))

    try:
        _, labels, _ = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        selected_indices = []
        # For each cluster, pick the one close to the center (or simply the first one found)
        # Here we just pick the first member of each cluster to ensure coverage
        seen_labels = set()
        for i, label in enumerate(labels.flatten()):
            if label not in seen_labels and len(seen_labels) < K:
                selected_indices.append(i)
                seen_labels.add(label)

        # If we missed some clusters (empty clusters?), fill up with randoms
        if len(selected_indices) < K:
            remaining = [i for i in range(len(candidates)) if i not in selected_indices]
            selected_indices.extend(remaining[:K - len(selected_indices)])

    except Exception as e:
        print(f"Clustering failed ({e}), falling back to uniform sampling.")
        # Fallback: Uniform sampling
        indices = np.linspace(0, len(candidates) - 1, K, dtype=int)
        selected_indices = indices.tolist()

    # 3. Sort back by original index to preserve timeline logic (roughly)
    selected_indices.sort(key=lambda i: candidates[i]["original_idx"])

    return [candidates[i]["original_b64"] for i in selected_indices]


def extract_key_frames(
    video_path: str,
    frame_limit: int = 10,
    sharpness_threshold: float = 40.0,
    max_resolution: int = 1024
) -> List[str]:
    """
    Selects diverse keyframes from a video using K-Means clustering.
    Instead of simple uniform sampling, it groups visually similar scenes
    and picks the sharpest image from each group to maximize information density.

    Args:
        video_path: Path to the video file.
        frame_limit: Maximum number of frames to extract (target K for clustering).
        sharpness_threshold: Variance of Laplacian threshold to ignore blurry frames.
        max_resolution: Max dimension (width/height) for resizing to control payload size.

    Returns:
        List of base64-encoded image strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Step 1: Gather candidates efficiently (~2 checks per second)
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

        # Skip blurry frames immediately to save processing time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < sharpness_threshold:
            continue

        # Resize large frames to prevent token/payload explosion
        h, w = frame.shape[:2]
        if max(h, w) > max_resolution:
            scale = max_resolution / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        candidates.append({
            "frame": frame,
            "timestamp": frame_idx / fps,
            "sharpness": sharpness,
            "signature": _get_frame_signature(frame)
        })

    cap.release()

    if not candidates:
        return []

    # Step 2: Intelligent Selection via K-Means
    # If we have fewer candidates than the limit, take them all.
    if len(candidates) <= frame_limit:
        final_candidates = candidates
    else:
        # Group frames by visual similarity
        data = np.array([c["signature"] for c in candidates], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Ensure K is valid (cannot be larger than the number of samples)
        K = min(frame_limit, len(candidates))

        # Run K-Means to find K distinct clusters of visual content
        _, labels, _ = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        selected_indices = []
        for i in range(K):
            # Find all frames belonging to this cluster
            cluster_indices = [idx for idx, label in enumerate(labels) if label == i]

            if not cluster_indices:
                continue

            # Pick the single sharpest frame from this cluster to represent the scene
            best_in_cluster = max(cluster_indices, key=lambda idx: candidates[idx]["sharpness"])
            selected_indices.append(best_in_cluster)

        final_candidates = [candidates[i] for i in selected_indices]

    # Step 3: Finalize (Sort by time, burn-in timestamps, convert to Base64)
    final_candidates.sort(key=lambda x: x["timestamp"])

    b64_list = []
    for item in final_candidates:
        frame = item["frame"]

        # Burn-in timestamp for the LLM context
        seconds = int(item["timestamp"])
        time_str = f"{seconds // 60:02d}:{seconds % 60:02d}"

        # Draw text with outline for readability on any background
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert to base64
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
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
