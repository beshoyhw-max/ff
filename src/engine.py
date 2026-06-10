"""
Forensic Engine — Orchestrates all 10 checks.
"""

import logging
from typing import List

from src.checks.base import BaseCheck
from src.checks.ela import ELACheck
from src.checks.noise import NoiseCheck
from src.checks.ai_detect import AIDetectCheck
from src.checks.copy_paste import CopyPasteCheck
from src.checks.splicing import SplicingCheck
from src.checks.jpeg_quant import JPEGQuantCheck
from src.checks.metadata import MetadataCheck
from src.checks.font_check import FontCheck
from src.checks.thumbnail import ThumbnailCheck
from src.checks.text_overlay import TextOverlayCheck

logger = logging.getLogger(__name__)


class ForensicEngine:
    """Orchestrates all forensic checks."""

    def __init__(self):
        self._checks: List[BaseCheck] = [
            ELACheck(),
            NoiseCheck(),
            AIDetectCheck(),
            CopyPasteCheck(),
            SplicingCheck(),
            JPEGQuantCheck(),
            MetadataCheck(),
            FontCheck(),
            ThumbnailCheck(),
            TextOverlayCheck(),
        ]

    def get_checks(self) -> List[BaseCheck]:
        return self._checks
