"""
Metadata & EXIF Forensics.

Inspects image EXIF and PDF metadata for tampering indicators:
- Software/creator tool analysis
- EXIF completeness and consistency
- Creation/modification date discrepancies
- Stripped metadata detection

Generalized from fraud V2 pdf_metadata.py (removed invoice-specific bias).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import create_info_card
from src.utils.image_loader import extract_exif

logger = logging.getLogger(__name__)


class MetadataCheck(BaseCheck):
    name = "Metadata / EXIF Forensics"
    check_id = "metadata"
    description = "Analyze file metadata and EXIF data for tampering indicators"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().metadata_config
        suspicious_software = config.get("suspicious_software", [])
        issues = []
        all_meta = {}
        visual_outputs = []

        if not file_path:
            card = create_info_card(
                {"Status": "No file path provided for metadata analysis"},
                title="Metadata",
            )
            path = self._save_evidence(card, "metadata", evidence_store, page_num)
            return self._make_pass_result(
                self.name, self.check_id,
                "No file path available for metadata analysis.",
                [VisualOutput(title="Metadata", image_path=path, description="N/A")], {},
            )

        ext = Path(file_path).suffix.lower()

        # Extract EXIF for images
        if ext in (".jpg", ".jpeg", ".tiff", ".tif"):
            exif = extract_exif(file_path)
            all_meta["exif"] = exif
            exif_issues = self._analyze_exif(exif, suspicious_software)
            issues.extend(exif_issues)
        else:
            all_meta["exif"] = {}

        # Extract PDF metadata
        if ext == ".pdf" and metadata and "pdf_metadata" in metadata:
            pdf_meta = metadata["pdf_metadata"]
            all_meta["pdf"] = pdf_meta
            pdf_issues = self._analyze_pdf_metadata(pdf_meta, suspicious_software)
            issues.extend(pdf_issues)
        else:
            all_meta["pdf"] = {}

        # Check for stripped/missing metadata
        if ext in (".jpg", ".jpeg"):
            if not all_meta.get("exif"):
                issues.append("EXIF data is completely absent — may have been stripped")

        # Build visual: metadata info card
        display_data = {}

        if all_meta.get("exif"):
            exif = all_meta["exif"]
            key_fields = [
                "Image Make", "Image Model", "Image Software",
                "EXIF DateTimeOriginal", "EXIF DateTimeDigitized",
                "Image DateTime", "GPS GPSLatitudeRef", "Image ImageWidth",
                "Image ImageLength", "Image XResolution",
            ]
            for k in key_fields:
                if k in exif:
                    display_data[k.replace("Image ", "").replace("EXIF ", "")] = exif[k]

            if not any(k in exif for k in key_fields):
                for k, v in list(exif.items())[:15]:
                    short_key = k.replace("Image ", "").replace("EXIF ", "")
                    display_data[short_key] = v

        if all_meta.get("pdf"):
            pdf = all_meta["pdf"]
            for k in ("title", "author", "creator", "producer", "creationDate", "modDate"):
                if pdf.get(k):
                    display_data[f"PDF {k}"] = pdf[k]

        if not display_data:
            display_data["Status"] = "No metadata found"

        # Add issues to display
        if issues:
            display_data["─── Issues ───"] = ""
            for i, issue in enumerate(issues):
                display_data[f"⚠️ Issue {i + 1}"] = issue

        card = create_info_card(display_data, title="File Metadata Analysis", width=700)
        path = self._save_evidence(card, "metadata", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Metadata",
            image_path=path,
            description="Extracted metadata and forensic findings",
        ))

        # Scoring
        if not issues:
            return self._make_pass_result(
                self.name, self.check_id,
                "Metadata appears clean and consistent.",
                visual_outputs, all_meta,
            )

        has_editor = any("editing software" in i.lower() or "photoshop" in i.lower() for i in issues)
        has_stripped = any("stripped" in i.lower() or "absent" in i.lower() for i in issues)
        has_date_issue = any("date" in i.lower() for i in issues)

        if has_editor:
            score = 70
            severity = Severity.HIGH
        elif has_stripped and has_date_issue:
            score = 50
            severity = Severity.MEDIUM
        elif has_stripped or has_date_issue:
            score = 35
            severity = Severity.MEDIUM
        else:
            score = 25
            severity = Severity.LOW

        summary = f"{len(issues)} metadata issue(s) found"
        detail = " | ".join(issues)

        return self._make_fail_result(
            self.name, self.check_id, score, severity,
            summary, detail, visual_outputs, all_meta,
        )

    def _analyze_exif(self, exif: Dict, suspicious: list) -> list:
        """Analyze EXIF data for issues."""
        issues = []
        software = exif.get("Image Software", "").lower()
        for tool in suspicious:
            if tool in software:
                issues.append(
                    f"Image was processed with editing software: {exif.get('Image Software')}"
                )
                break

        original = exif.get("EXIF DateTimeOriginal", "")
        digitized = exif.get("EXIF DateTimeDigitized", "")
        modified = exif.get("Image DateTime", "")

        if original and modified and original != modified:
            issues.append(
                f"Original date ({original}) differs from modification date ({modified})"
            )

        has_gps = any("GPS" in k for k in exif)
        if has_gps and software and any(t in software for t in suspicious):
            issues.append("GPS data present but image was processed with editing software")

        return issues

    def _analyze_pdf_metadata(self, meta: Dict, suspicious: list) -> list:
        """Analyze PDF metadata."""
        issues = []
        creator = (meta.get("creator") or "").lower()
        producer = (meta.get("producer") or "").lower()

        for tool in suspicious:
            if tool in creator or tool in producer:
                issues.append(
                    f"PDF created/produced with editing tool: "
                    f"creator='{meta.get('creator')}', producer='{meta.get('producer')}'"
                )
                break

        creation = meta.get("creationDate", "")
        mod = meta.get("modDate", "")
        if creation and mod and creation != mod:
            issues.append(f"PDF modified after creation: created={creation}, modified={mod}")

        if not creator and not producer:
            issues.append("PDF creator/producer metadata is missing")

        return issues
