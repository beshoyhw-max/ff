"""
Evidence Store — Manages disk-based storage for page renders and check evidence.

Two top-level directories per analysis session:
  data/{uuid}/pages/        → rendered page images
  evidence/{uuid}/{check}/  → check visual evidence
"""

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from PIL import Image

logger = logging.getLogger(__name__)


class EvidenceStore:
    """Manages on-disk storage for analysis session artifacts."""

    def __init__(self, project_root: str):
        self._uuid = uuid4().hex[:12]
        self._project_root = Path(project_root)
        self._data_dir = self._project_root / "data" / self._uuid / "pages"
        self._evidence_dir = self._project_root / "evidence" / self._uuid

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EvidenceStore session %s — data: %s, evidence: %s",
            self._uuid, self._data_dir, self._evidence_dir,
        )

    @property
    def session_id(self) -> str:
        return self._uuid

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def evidence_dir(self) -> Path:
        return self._evidence_dir

    def save_page(self, image: Image.Image, page_num: int) -> str:
        """Save a rendered page image to data/{uuid}/pages/page_NNN.png."""
        path = self._data_dir / f"page_{page_num:03d}.png"
        image.save(str(path), format="PNG")
        logger.debug("Saved page %d → %s", page_num, path)
        return str(path)

    def save_evidence(
        self,
        image: Image.Image,
        check_id: str,
        name: str,
        page_num: int = 0,
    ) -> str:
        """Save check evidence image to evidence/{uuid}/{check_id}/page_NNN_{name}.png."""
        check_dir = self._evidence_dir / check_id
        check_dir.mkdir(parents=True, exist_ok=True)

        filename = f"page_{page_num:03d}_{name}.png"
        path = check_dir / filename
        image.save(str(path), format="PNG")
        logger.debug("Saved evidence %s/%s → %s", check_id, filename, path)
        return str(path)

    def cleanup(self) -> None:
        """Remove this session's data and evidence directories."""
        data_parent = self._data_dir.parent  # data/{uuid}/
        try:
            shutil.rmtree(str(data_parent), ignore_errors=True)
            shutil.rmtree(str(self._evidence_dir), ignore_errors=True)
            logger.info("Cleaned up session %s", self._uuid)
        except Exception as e:
            logger.warning("Cleanup failed for session %s: %s", self._uuid, e)

    @staticmethod
    def cleanup_all(project_root: str) -> None:
        """Remove all data/ and evidence/ session folders."""
        root = Path(project_root)
        for folder in ("data", "evidence"):
            target = root / folder
            if target.exists():
                shutil.rmtree(str(target), ignore_errors=True)
                logger.info("Cleaned up all %s/", folder)
