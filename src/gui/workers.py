"""
QThread workers for background analysis.

Multi-page support: For PDFs each page is rendered, saved to disk,
analyzed with all 10 checks, and the page image is released from memory.
Signals carry paths (not PIL objects) so the GUI can lazy-load.
"""

import time
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, Signal

from PIL import Image

from src.models import (
    AnalysisResult,
    CheckResult,
    FileType,
    PageResult,
    Verdict,
)
from src.utils.evidence_store import EvidenceStore


class AnalysisWorker(QThread):
    """Runs all forensic checks in a background thread."""

    # ── Signals ──
    # page_started(page_num, total_pages)
    page_started = Signal(int, int)
    # check_started(check_name)
    check_started = Signal(str)
    # check_completed(CheckResult, page_num)
    check_completed = Signal(object, int)
    # page_completed(PageResult)
    page_completed = Signal(object)
    # all_completed(AnalysisResult)
    all_completed = Signal(object)
    # progress(current_step, total_steps)  — flat step counter
    progress = Signal(int, int)
    # error(message)
    error = Signal(str)

    def __init__(
        self,
        file_path: str,
        file_type: FileType,
        metadata: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._file_path = file_path
        self._file_type = file_type
        self._metadata = metadata
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        from src.engine import ForensicEngine
        from src.config import Config
        from src.utils.image_loader import load_image, load_all_pdf_pages, extract_pdf_metadata

        try:
            config = Config.get()
            engine = ForensicEngine()
            checks = engine.get_checks()
            store = EvidenceStore(str(config.project_root))

            start = time.perf_counter()

            # ── Determine pages and extract file-level metadata ──
            if self._file_type == FileType.PDF:
                # Extract PDF metadata once for the whole file
                self._metadata["pdf_metadata"] = extract_pdf_metadata(self._file_path)
                page_images = load_all_pdf_pages(self._file_path)
            else:
                image, _, meta = load_image(self._file_path)
                page_images = [image]
                # Merge loader metadata into ours
                self._metadata.update(meta)

            total_pages = len(page_images)
            total_steps = total_pages * len(checks)
            step = 0

            page_results: List[PageResult] = []

            for page_idx, page_img in enumerate(page_images):
                if self._cancelled:
                    break

                self.page_started.emit(page_idx, total_pages)

                # Save rendered page to disk
                page_path = store.save_page(page_img, page_idx)

                page_result = PageResult(
                    page_number=page_idx,
                    page_image_path=page_path,
                    page_size=page_img.size,
                )

                # Run each check on this page
                for check in checks:
                    if self._cancelled:
                        break

                    self.check_started.emit(check.name)
                    self.progress.emit(step, total_steps)

                    result = check.safe_run(
                        page_img,
                        file_path=self._file_path,
                        metadata=self._metadata,
                        evidence_store=store,
                        page_num=page_idx,
                    )
                    page_result.checks.append(result)
                    self.check_completed.emit(result, page_idx)
                    step += 1

                # Page done — compute page score
                page_result.compute_overall(config.weights)
                page_results.append(page_result)
                self.page_completed.emit(page_result)

                # Release page image from memory
                del page_img

            # Clean up the list too
            del page_images

            elapsed = time.perf_counter() - start

            if not self._cancelled:
                analysis = AnalysisResult(
                    file_path=self._file_path,
                    file_name=Path(self._file_path).name,
                    file_type=self._file_type,
                    pages=page_results,
                    analysis_time_seconds=round(elapsed, 2),
                    session_id=store.session_id,
                )
                analysis.compute_overall(config.weights)
                self.progress.emit(total_steps, total_steps)
                self.all_completed.emit(analysis)

        except Exception as e:
            self.error.emit(str(e))
