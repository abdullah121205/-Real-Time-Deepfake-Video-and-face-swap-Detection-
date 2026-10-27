"""
Face-swap forensic analyzer module.
Detects physical and digital manipulation artifacts characteristic of face swapping:
1. Blending boundary edge discontinuity (Face X-Ray principle).
2. High-frequency 2D-FFT spectral distribution anomalies.
3. Inner vs outer facial region color/illumination disparity.
"""

from typing import Dict, Tuple
import cv2
import numpy as np


class FaceSwapDetector:
    """
    Forensic analyzer specialized in identifying face-swapping operations
    by inspecting boundary blending seams, frequency power spectra, and tone consistency.
    """

    def __init__(
        self,
        boundary_weight: float = 0.40,
        frequency_weight: float = 0.35,
        color_weight: float = 0.25
    ):
        """
        Args:
            boundary_weight: Importance of perimeter edge discontinuities.
            frequency_weight: Importance of FFT high-frequency spectral roll-off.
            color_weight: Importance of inner-vs-outer color tone disparity.
        """
        self.w_boundary = boundary_weight
        self.w_freq = frequency_weight
        self.w_color = color_weight

    def analyze_face(self, face_bgr: np.ndarray) -> Dict[str, float]:
        """
        Runs forensic checks across a cropped face region.

        Args:
            face_bgr: Cropped BGR face image with perimeter margin.

        Returns:
            Dictionary with composite 'swap_score' and individual forensic metrics:
            - 'swap_score': Overall face-swap likelihood [0.0 - 1.0].
            - 'boundary_anomaly': Blending seam / gradient discontinuity score.
            - 'frequency_anomaly': High-frequency spectral loss score.
            - 'color_disparity': Tone difference between face core and boundary.
        """
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
            return {
                "swap_score": 0.0,
                "boundary_anomaly": 0.0,
                "frequency_anomaly": 0.0,
                "color_disparity": 0.0
            }

        # 1. Inspect Boundary Blending Seams (Laplacian Edge Variance)
        boundary_score = self._detect_boundary_artifacts(face_bgr)

        # 2. Inspect 2D FFT Frequency Distribution
        freq_score = self._analyze_frequency_spectrum(face_bgr)

        # 3. Inspect Color / Illumination Uniformity
        color_score = self._analyze_color_disparity(face_bgr)

        # Weighted composite score
        composite = (
            self.w_boundary * boundary_score +
            self.w_freq * freq_score +
            self.w_color * color_score
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return {
            "swap_score": composite,
            "boundary_anomaly": float(boundary_score),
            "frequency_anomaly": float(freq_score),
            "color_disparity": float(color_score)
        }

    def _detect_boundary_artifacts(self, img_bgr: np.ndarray) -> float:
        """
        Detects artificial edge blurriness or sudden sharp gradient jumps
        along the perimeter where swapped faces are blended into target heads.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Define outer ring mask (outer 15% margin) and inner center mask (middle 60%)
        outer_mask = np.ones((h, w), dtype=np.uint8)
        inner_y1, inner_y2 = int(h * 0.20), int(h * 0.80)
        inner_x1, inner_x2 = int(w * 0.20), int(w * 0.80)
        outer_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        # Compute gradient magnitude using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        # In natural faces, gradient transition across skin is continuous.
        # Swapping masks produce either sharp blending lines or over-smoothed blurred rings.
        outer_mag_mean = np.mean(mag[outer_mask == 1]) + 1e-6
        inner_mag_mean = np.mean(mag[outer_mask == 0]) + 1e-6

        ratio = outer_mag_mean / inner_mag_mean
        # Extreme ratio indicates boundary artifact
        anomaly = np.abs(ratio - 1.0)
        # Normalize into [0, 1] sigmoid range
        score = 1.0 / (1.0 + np.exp(-3.0 * (anomaly - 0.5)))
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_frequency_spectrum(self, img_bgr: np.ndarray) -> float:
        """
        Computes the 2D Fast Fourier Transform magnitude spectrum.
        Deep generative networks (autoencoders/GANs/diffusion) exhibit unnatural
        suppression of high-frequency textures (pores, sensor noise, hair strands).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Standardize size for consistent frequency analysis
        resized = cv2.resize(gray, (128, 128)).astype(np.float32)

        # 2D Fast Fourier Transform with DC shift to center
        dft = np.fft.fft2(resized)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.log(np.abs(dft_shift) + 1.0)

        # Divide into low-frequency core and high-frequency perimeter
        center_y, center_x = 64, 64
        y, x = np.ogrid[:128, :128]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        low_freq_mask = dist_from_center <= 24
        high_freq_mask = dist_from_center > 40

        low_power = np.mean(magnitude[low_freq_mask]) + 1e-6
        high_power = np.mean(magnitude[high_freq_mask]) + 1e-6

        # Ratio of high to low frequency power
        hf_ratio = high_power / low_power

        # Typical natural face crops have hf_ratio around 0.55 - 0.70.
        # Heavy generative smoothing drops it lower, or checkerboard artifacts spike it.
        expected_ratio = 0.62
        deviation = np.abs(hf_ratio - expected_ratio)
        score = 1.0 / (1.0 + np.exp(-12.0 * (deviation - 0.12)))
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_color_disparity(self, img_bgr: np.ndarray) -> float:
        """
        Measures chromatic disparity in LAB color space between the central face
        and outer bounding context (jawline/hair/neck).
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]

        inner = lab[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        outer_mask = np.ones((h, w), dtype=bool)
        outer_mask[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)] = False

        inner_a_mean = np.mean(inner[:, :, 1])
        inner_b_mean = np.mean(inner[:, :, 2])

        outer_a_mean = np.mean(lab[:, :, 1][outer_mask])
        outer_b_mean = np.mean(lab[:, :, 2][outer_mask])

        # Euclidean chromatic distance in A/B channels
        chroma_dist = np.sqrt((inner_a_mean - outer_a_mean) ** 2 + (inner_b_mean - outer_b_mean) ** 2)

        # Higher chromatic divergence indicates inconsistent pasted face mask
        norm_dist = chroma_dist / 40.0
        score = 1.0 / (1.0 + np.exp(-5.0 * (norm_dist - 0.35)))
        return float(np.clip(score, 0.0, 1.0))
