"""
AI-Generated Image Detection.

Detects AI-generated images (GAN, Stable Diffusion, Midjourney) using:
- DCT spectral signature analysis with visualization
- JPEG ghost curves
- GLCM texture descriptors
- GAN checkerboard artifact detection

Enhanced from fraud V2: adds DCT spectrum visualization, GAN pattern detection.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import numpy_to_pil_heatmap, create_info_card

logger = logging.getLogger(__name__)


class AIDetectCheck(BaseCheck):
    name = "AI Generation Detection"
    check_id = "ai_detect"
    description = "Detect AI-generated images via frequency domain and texture analysis"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().ai_detection
        visual_outputs = []

        # 1. DCT spectrum analysis + visualization
        dct_score, dct_ev, dct_visual = self._analyze_dct(image)
        path = self._save_evidence(dct_visual, "dct_spectrum", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="DCT Spectrum",
            image_path=path,
            description=(
                f"DCT frequency spectrum (log magnitude). "
                f"AI images show unusual energy distribution. "
                f"Freq ratio: {dct_ev.get('freq_ratio', 0):.4f}"
            ),
        ))

        # 2. JPEG ghost analysis
        ghost_score, ghost_ev, ghost_visual = self._analyze_ghosts(image)
        path = self._save_evidence(ghost_visual, "jpeg_ghosts", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="JPEG Ghosts",
            image_path=path,
            description=(
                f"JPEG re-compression error curve. "
                f"Monotonic: {ghost_ev.get('is_monotonic', False)}, "
                f"Variance: {ghost_ev.get('diff_variance', 0):.4f}"
            ),
        ))

        # 3. GLCM texture analysis
        glcm_score, glcm_ev, glcm_visual = self._analyze_glcm(image)
        path = self._save_evidence(glcm_visual, "glcm_texture", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Texture Analysis",
            image_path=path,
            description=(
                f"GLCM texture properties. "
                f"Homogeneity: {glcm_ev.get('homogeneity', 0):.3f}, "
                f"Contrast: {glcm_ev.get('contrast', 0):.3f}"
            ),
        ))

        # 4. GAN checkerboard detection
        gan_score, gan_ev = self._detect_gan_artifacts(image)

        # Weighted combination
        combined = int(
            dct_score * 0.30 + ghost_score * 0.25
            + glcm_score * 0.25 + gan_score * 0.20
        )

        evidence = {
            "dct": {"score": dct_score, **dct_ev},
            "jpeg_ghost": {"score": ghost_score, **ghost_ev},
            "glcm": {"score": glcm_score, **glcm_ev},
            "gan_artifacts": {"score": gan_score, **gan_ev},
            "combined_score": combined,
        }

        if combined >= 65:
            severity = Severity.CRITICAL
            summary = f"High probability AI-generated (score: {combined})"
            detail = (
                f"Strong AI-generation indicators: "
                f"DCT={dct_score}, Ghost={ghost_score}, GLCM={glcm_score}, GAN={gan_score}."
            )
        elif combined >= 40:
            severity = Severity.HIGH
            summary = f"Moderate AI-generation indicators (score: {combined})"
            detail = (
                f"Moderate AI indicators: "
                f"DCT={dct_score}, Ghost={ghost_score}, GLCM={glcm_score}, GAN={gan_score}."
            )
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"No significant AI-generation artifacts. Score: {combined}/100.",
                visual_outputs, evidence,
            )

        return self._make_fail_result(
            self.name, self.check_id, combined, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _analyze_dct(self, image: Image.Image) -> Tuple[int, Dict, Image.Image]:
        """DCT spectrum analysis with visualization."""
        gray = np.array(image.convert("L"), dtype=np.float32)
        gray = cv2.resize(gray, (512, 512))

        dct = cv2.dct(gray)
        h, w = dct.shape

        # Log magnitude spectrum for visualization
        magnitude = np.abs(dct)
        log_mag = np.log1p(magnitude)
        log_mag_normalized = (log_mag / log_mag.max() * 255).astype(np.uint8)
        spectrum_img = numpy_to_pil_heatmap(log_mag_normalized, cv2.COLORMAP_INFERNO)

        # Energy distribution
        low_freq = np.abs(dct[:h // 4, :w // 4])
        mid_freq = np.abs(dct[h // 4:h // 2, w // 4:w // 2])
        high_freq = np.abs(dct[h // 2:, w // 2:])

        low_energy = float(np.mean(low_freq))
        mid_energy = float(np.mean(mid_freq))
        high_energy = float(np.mean(high_freq))

        freq_ratio = (mid_energy + high_energy) / low_energy if low_energy > 0 else 0

        if freq_ratio > 0.25:
            score = min(100, int(60 + freq_ratio * 100))
        elif freq_ratio > 0.15:
            score = int(30 + freq_ratio * 100)
        else:
            score = 0

        evidence = {
            "low_energy": round(low_energy, 4),
            "mid_energy": round(mid_energy, 4),
            "high_energy": round(high_energy, 4),
            "freq_ratio": round(freq_ratio, 4),
        }

        return score, evidence, spectrum_img

    def _analyze_ghosts(
        self, image: Image.Image
    ) -> Tuple[int, Dict, Image.Image]:
        """JPEG ghost analysis with error curve visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original = np.array(image.convert("RGB"), dtype=np.float32)
        rgb = image.convert("RGB")
        qualities = [50, 60, 70, 80, 85, 90, 95]
        errors = {}

        for q in qualities:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
            error = float(np.mean(np.abs(original - recomp)))
            errors[q] = round(error, 4)

        error_vals = list(errors.values())
        diffs = [error_vals[i] - error_vals[i + 1] for i in range(len(error_vals) - 1)]
        is_monotonic = all(d >= -0.5 for d in diffs)
        diff_variance = float(np.var(diffs)) if diffs else 0

        # Plot error curve
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#161b22")
        ax.set_facecolor("#0d1117")
        ax.plot(qualities, error_vals, "o-", color="#00d2ff", linewidth=2, markersize=6)
        ax.set_xlabel("JPEG Quality", color="#8b949e", fontsize=10)
        ax.set_ylabel("Mean Error", color="#8b949e", fontsize=10)
        ax.set_title("JPEG Ghost Error Curve", color="#e6edf3", fontsize=12)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, alpha=0.2, color="#30363d")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        plot_img = Image.open(buf).convert("RGB")

        if is_monotonic and diff_variance < 1.0:
            score = min(100, int(50 + (1.0 - diff_variance) * 50))
        elif diff_variance < 2.0:
            score = int(20 + (2.0 - diff_variance) * 15)
        else:
            score = 0

        evidence = {
            "quality_errors": errors,
            "is_monotonic": is_monotonic,
            "diff_variance": round(diff_variance, 4),
        }

        return score, evidence, plot_img

    def _analyze_glcm(self, image: Image.Image) -> Tuple[int, Dict, Image.Image]:
        """GLCM texture analysis with info card visualization."""
        from skimage.feature import graycomatrix, graycoprops

        gray = np.array(image.convert("L"))
        gray = cv2.resize(gray, (256, 256))
        gray_q = (gray // 4).astype(np.uint8)

        distances = [1, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            gray_q, distances=distances, angles=angles,
            levels=64, symmetric=True, normed=True
        )

        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        correlation = float(np.mean(graycoprops(glcm, "correlation")))

        suspicion = 0
        reasons = []
        if homogeneity > 0.7:
            suspicion += 30
            reasons.append("High homogeneity (AI-like smoothness)")
        if contrast < 5.0:
            suspicion += 20
            reasons.append("Low contrast (unnaturally smooth)")
        if energy > 0.1:
            suspicion += 25
            reasons.append("High energy (repetitive texture)")
        if correlation > 0.95:
            suspicion += 25
            reasons.append("High correlation (synthetic pattern)")

        score = min(100, suspicion)

        # Info card visualization
        card_data = {
            "Contrast": f"{contrast:.4f}",
            "Homogeneity": f"{homogeneity:.4f} {'⚠️' if homogeneity > 0.7 else '✅'}",
            "Energy": f"{energy:.4f} {'⚠️' if energy > 0.1 else '✅'}",
            "Correlation": f"{correlation:.4f} {'⚠️' if correlation > 0.95 else '✅'}",
            "Suspicion Score": f"{score}/100",
        }
        if reasons:
            card_data["Flags"] = ", ".join(reasons)

        card_img = create_info_card(card_data, title="GLCM Texture Properties")

        evidence = {
            "contrast": round(contrast, 4),
            "homogeneity": round(homogeneity, 4),
            "energy": round(energy, 4),
            "correlation": round(correlation, 4),
            "reasons": reasons,
        }

        return score, evidence, card_img

    def _detect_gan_artifacts(self, image: Image.Image) -> Tuple[int, Dict]:
        """Detect GAN checkerboard artifacts in frequency domain."""
        gray = np.array(image.convert("L"), dtype=np.float32)
        gray = cv2.resize(gray, (256, 256))

        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        log_mag = np.log1p(magnitude)

        # GAN artifacts appear as periodic peaks
        h, w = log_mag.shape
        center_h, center_w = h // 2, w // 2

        # Mask out the center (low frequency) and edges
        mask = np.zeros_like(log_mag, dtype=bool)
        for y in range(h):
            for x in range(w):
                r = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                if 20 < r < min(center_h, center_w) - 10:
                    mask[y, x] = True

        mid_freq = log_mag[mask]
        if len(mid_freq) == 0:
            return 0, {"insufficient_data": True}

        mean_mag = float(np.mean(mid_freq))
        std_mag = float(np.std(mid_freq))
        max_mag = float(np.max(mid_freq))

        # Strong peaks relative to mean = checkerboard artifacts
        peak_ratio = (max_mag - mean_mag) / std_mag if std_mag > 0 else 0

        if peak_ratio > 8.0:
            score = min(100, int(50 + peak_ratio * 5))
        elif peak_ratio > 5.0:
            score = int(20 + peak_ratio * 5)
        else:
            score = 0

        evidence = {
            "peak_ratio": round(peak_ratio, 3),
            "mean_magnitude": round(mean_mag, 3),
            "max_magnitude": round(max_mag, 3),
        }

        return score, evidence
