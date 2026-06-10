"""
PDF Report Generator.

Generates a comprehensive PDF report with all check results and evidence images.
Multi-page documents get per-page sections.
Evidence images are loaded from disk paths.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models import AnalysisResult, CheckResult, PageResult, Severity, Verdict

logger = logging.getLogger(__name__)


def generate_report(result: AnalysisResult, output_path: str) -> None:
    """Generate a PDF report from analysis results."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#e6edf3"),
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#00d2ff"),
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#c9d1d9"),
        spaceAfter=4,
        leading=14,
    )
    detail_style = ParagraphStyle(
        "Detail",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#8b949e"),
        spaceAfter=4,
        leading=12,
    )

    elements = []

    # ── Title Page ──
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph("🔍 Forgery Detection Report", title_style))
    elements.append(Spacer(1, 0.5 * cm))

    verdict_color = {
        Verdict.CLEAN: "#00ff88",
        Verdict.SUSPICIOUS: "#ffaa00",
        Verdict.TAMPERED: "#ff2244",
    }.get(result.overall_verdict, "#e6edf3")

    elements.append(Paragraph(
        f'<font color="{verdict_color}" size="18">'
        f'{result.overall_verdict.value}</font>',
        body_style,
    ))
    elements.append(Paragraph(
        f'Overall Score: <font color="#00d2ff">{result.overall_score}/100</font>',
        body_style,
    ))
    elements.append(Spacer(1, 0.5 * cm))

    # File info table
    info_data = [
        ["File", result.file_name],
        ["Type", result.file_type.value.upper()],
        ["Pages", str(result.page_count)],
        ["Analyzed", result.timestamp.strftime("%Y-%m-%d %H:%M:%S")],
        ["Duration", f"{result.analysis_time_seconds:.1f} seconds"],
        ["Total Checks", str(len(result.checks))],
        ["Triggered", str(len(result.triggered_checks))],
    ]

    info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
    info_table.setStyle(TableStyle([
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#8b949e")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#e6edf3")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#21262d")),
    ]))
    elements.append(info_table)
    elements.append(PageBreak())

    # ── Per-Page Results ──
    for page in result.pages:
        elements.extend(
            _build_page_section(page, heading_style, body_style, detail_style)
        )

    # Build the PDF
    doc.build(elements)
    logger.info(f"PDF report generated: {output_path}")


def _build_page_section(
    page: PageResult,
    heading_style: ParagraphStyle,
    body_style: ParagraphStyle,
    detail_style: ParagraphStyle,
) -> list:
    """Build elements for a single page section."""
    elements = []

    # Page header
    elements.append(Paragraph(
        f"📄 Page {page.page_number + 1} — "
        f'{page.verdict_icon} {page.overall_verdict.value} '
        f'(Score: {page.overall_score}/100)',
        heading_style,
    ))

    # Page image
    if page.page_image_path and Path(page.page_image_path).exists():
        rl_img = _path_to_reportlab(
            page.page_image_path, max_width=14 * cm, max_height=8 * cm
        )
        if rl_img:
            elements.append(rl_img)
            elements.append(Spacer(1, 0.3 * cm))

    # Summary table for this page
    sorted_checks = sorted(
        page.checks,
        key=lambda c: (-c.severity.rank, -c.score),
    )

    summary_data = [["Check", "Score", "Severity", "Status"]]
    for check in sorted_checks:
        status = "⚠️ TRIGGERED" if check.triggered else "✅ CLEAN"
        summary_data.append([
            check.name,
            f"{check.score}/100",
            check.severity.value.upper(),
            status,
        ])

    summary_table = Table(
        summary_data,
        colWidths=[6 * cm, 3 * cm, 3 * cm, 4 * cm],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21262d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#e6edf3")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#c9d1d9")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#21262d")),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
    ]))
    elements.append(summary_table)
    elements.append(PageBreak())

    # Individual check details
    for check in sorted_checks:
        elements.extend(
            _build_check_section(check, heading_style, body_style, detail_style)
        )
        elements.append(PageBreak())

    return elements


def _build_check_section(
    check: CheckResult,
    heading_style: ParagraphStyle,
    body_style: ParagraphStyle,
    detail_style: ParagraphStyle,
) -> list:
    """Build elements for a single check section."""
    elements = []

    status_icon = check.status_icon
    sev_color = check.severity.color_hex

    elements.append(Paragraph(
        f'{status_icon} {check.name}',
        heading_style,
    ))

    elements.append(Paragraph(
        f'Score: <font color="{sev_color}">{check.score}/100</font> | '
        f'Severity: <font color="{sev_color}">{check.severity.value.upper()}</font> | '
        f'Duration: {check.duration_ms:.0f}ms',
        body_style,
    ))

    elements.append(Paragraph(check.summary, body_style))
    elements.append(Spacer(1, 0.3 * cm))

    # Evidence images — load from disk
    for vo in check.visual_outputs:
        elements.append(Paragraph(
            f'<font color="#00d2ff">{vo.title}</font>',
            body_style,
        ))

        if vo.image_path and Path(vo.image_path).exists():
            rl_img = _path_to_reportlab(
                vo.image_path, max_width=16 * cm, max_height=10 * cm
            )
            if rl_img:
                elements.append(rl_img)

        if vo.description:
            elements.append(Paragraph(vo.description, detail_style))

        elements.append(Spacer(1, 0.3 * cm))

    # Detail text
    if check.detail:
        elements.append(Paragraph(
            f'<b>Details:</b> {check.detail}',
            detail_style,
        ))

    return elements


def _path_to_reportlab(
    image_path: str,
    max_width: float = 14 * cm,
    max_height: float = 8 * cm,
) -> Optional[RLImage]:
    """Load image from disk path and convert to ReportLab Image."""
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        img.close()

        scale_w = max_width / img_w
        scale_h = max_height / img_h
        scale = min(scale_w, scale_h, 1.0)

        rl_img = RLImage(image_path, width=img_w * scale, height=img_h * scale)
        return rl_img
    except Exception as e:
        logger.warning(f"Failed to load image for PDF: {e}")
        return None
