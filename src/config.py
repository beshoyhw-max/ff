"""
Configuration loader for the Universal Forgery Detection system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_DEFAULT_CONFIG = {
    "ela": {
        "quality_levels": [60, 70, 75, 80, 85, 90, 95],
        "scale_factor": 15,
        "region_threshold": 60,
        "consensus_min_levels": 3,
        "min_contour_area": 200,
    },
    "noise": {
        "wavelet_levels": 4,
        "wavelet_name": "db4",
        "block_size": 64,
        "outlier_z_threshold": 2.0,
        "min_blocks": 8,
        "prnu_window_size": 128,
        "prnu_step": 64,
    },
    "ai_detection": {
        "dct_ai_freq_ratio_threshold": 0.15,
        "glcm_homogeneity_threshold": 0.7,
        "glcm_contrast_threshold": 5.0,
        "ghost_monotonic_variance_threshold": 1.0,
    },
    "copy_paste": {
        "keypoint_method": "sift",
        "max_keypoints": 5000,
        "match_ratio_threshold": 0.75,
        "min_cluster_size": 8,
        "cluster_eps": 30,
        "block_size": 16,
        "block_overlap": 8,
        "block_dct_match_threshold": 0.95,
    },
    "splicing": {
        "block_size": 64,
        "color_temp_z_threshold": 2.5,
        "edge_gradient_threshold": 100,
        "noise_mismatch_threshold": 2.0,
    },
    "jpeg_quantization": {
        "test_qualities": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "ghost_block_size": 8,
    },
    "metadata": {
        "suspicious_software": [
            "photoshop", "gimp", "illustrator", "affinity",
            "pixelmator", "paint.net",
        ],
    },
    "font": {
        "min_text_regions": 5,
        "stroke_outlier_z": 2.5,
        "edge_hist_threshold": 0.5,
        "ssim_outlier_z": 2.0,
        "text_density_threshold": 0.02,
    },
    "thumbnail": {
        "ssim_mismatch_threshold": 0.9,
        "pixel_diff_threshold": 15,
    },
    "text_overlay": {
        "noise_ratio_threshold": 2.0,
        "min_text_height": 10,
        "edge_sharpness_threshold": 50,
    },
    "weights": {
        "ela": 15,
        "noise": 10,
        "ai_detect": 12,
        "copy_paste": 15,
        "splicing": 12,
        "jpeg_quant": 8,
        "metadata": 5,
        "font_check": 8,
        "thumbnail": 5,
        "text_overlay": 10,
    },
    "gui": {
        "default_mode": "simple",
        "theme": "dark",
        "max_image_dimension": 4096,
        "window_width": 1400,
        "window_height": 900,
    },
    "export": {
        "output_dir": "reports",
        "include_raw_evidence": False,
    },
    "storage": {
        "data_dir": "data",
        "evidence_dir": "evidence",
    },
}


class Config:
    """Singleton configuration manager."""

    _instance: Optional["Config"] = None
    _data: Dict[str, Any] = {}

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = Config()
            cls._instance._load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _load(self) -> None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            self._data = self._deep_merge(_DEFAULT_CONFIG, file_data)
        else:
            self._data = _DEFAULT_CONFIG.copy()

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._data.get(section, {})

    @property
    def ela(self) -> Dict[str, Any]:
        return self._data.get("ela", {})

    @property
    def noise(self) -> Dict[str, Any]:
        return self._data.get("noise", {})

    @property
    def ai_detection(self) -> Dict[str, Any]:
        return self._data.get("ai_detection", {})

    @property
    def copy_paste(self) -> Dict[str, Any]:
        return self._data.get("copy_paste", {})

    @property
    def splicing(self) -> Dict[str, Any]:
        return self._data.get("splicing", {})

    @property
    def jpeg_quantization(self) -> Dict[str, Any]:
        return self._data.get("jpeg_quantization", {})

    @property
    def metadata_config(self) -> Dict[str, Any]:
        return self._data.get("metadata", {})

    @property
    def font(self) -> Dict[str, Any]:
        return self._data.get("font", {})

    @property
    def thumbnail_config(self) -> Dict[str, Any]:
        return self._data.get("thumbnail", {})

    @property
    def text_overlay(self) -> Dict[str, Any]:
        return self._data.get("text_overlay", {})

    @property
    def weights(self) -> Dict[str, int]:
        return self._data.get("weights", {})

    @property
    def gui(self) -> Dict[str, Any]:
        return self._data.get("gui", {})

    @property
    def export(self) -> Dict[str, Any]:
        return self._data.get("export", {})

    @property
    def storage(self) -> Dict[str, Any]:
        return self._data.get("storage", {})

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def reports_dir(self) -> Path:
        p = Path(self.export.get("output_dir", "reports"))
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        p.mkdir(parents=True, exist_ok=True)
        return p
