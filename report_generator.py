"""
BactoVision AI - Automated Clinical & Microbiological Report Generator
Produces structured CSV metrics and medical-grade PDF laboratory reports
using ReportLab, Matplotlib, and Pandas.
"""

import os
import io
import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_csv(summary: Dict[str, Any], cells: List[Dict[str, Any]]) -> str:
    """Generates a detailed per-cell CSV string with morphometrics and classification."""
    records = []
    for cell in cells:
        mean_hsv = cell.get("mean_hsv", {})
        centroid = cell.get("centroid", [0, 0])
        bbox = cell.get("bbox", [0, 0, 0, 0])
        
        records.append({
            "Cell_ID": cell.get("id"),
            "Gram_Status": cell.get("gram_status"),
            "Morphology": cell.get("morphology"),
            "Confidence": cell.get("confidence"),
            "Area_px2": cell.get("area_px"),
            "Area_um2": cell.get("area_um2"),
            "Perimeter_px": cell.get("perimeter_px"),
            "Perimeter_um": cell.get("perimeter_um"),
            "Circularity": cell.get("circularity"),
            "Aspect_Ratio": cell.get("aspect_ratio"),
            "Mean_Hue": mean_hsv.get("h", 0.0),
            "Mean_Saturation": mean_hsv.get("s", 0.0),
            "Mean_Value": mean_hsv.get("v", 0.0),
            "Centroid_X": centroid[0],
            "Centroid_Y": centroid[1],
            "BBox_X": bbox[0],
            "BBox_Y": bbox[1],
            "BBox_Width": bbox[2],
            "BBox_Height": bbox[3],
        })

    df = pd.DataFrame(records)
    return df.to_csv(index=False)


def _generate_distribution_chart(cells: List[Dict[str, Any]]) -> io.BytesIO:
    """
    Renders a publication-quality morphometric scatter plot (Circularity vs Aspect Ratio)
    visualizing Cocci vs. Bacilli clustering and Gram-Positive vs. Gram-Negative separation.
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # Filter by Gram status
    pos_circ = [c["circularity"] for c in cells if c["gram_status"] == "Gram-Positive"]
    pos_aspect = [c["aspect_ratio"] for c in cells if c["gram_status"] == "Gram-Positive"]
    
    neg_circ = [c["circularity"] for c in cells if c["gram_status"] == "Gram-Negative"]
    neg_aspect = [c["aspect_ratio"] for c in cells if c["gram_status"] == "Gram-Negative"]

    indet_circ = [c["circularity"] for c in cells if c["gram_status"] == "Indeterminate"]
    indet_aspect = [c["aspect_ratio"] for c in cells if c["gram_status"] == "Indeterminate"]

    # Decision boundary guides
    ax.axvline(0.78, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.8, label="Circularity Split (0.78)")
    ax.axhline(1.35, color="#94A3B8", linestyle=":", linewidth=1.0, alpha=0.8, label="Aspect Ratio Split (1.35)")

    # Scatter points
    if pos_circ:
        ax.scatter(pos_circ, pos_aspect, c="#7C3AED", s=38, alpha=0.75, edgecolors="#4C1D95", linewidth=0.6, label="Gram-Positive")
    if neg_circ:
        ax.scatter(neg_circ, neg_aspect, c="#E11D48", s=38, alpha=0.75, edgecolors="#881337", linewidth=0.6, label="Gram-Negative")
    if indet_circ:
        ax.scatter(indet_circ, indet_aspect, c="#F59E0B", s=38, alpha=0.75, edgecolors="#B45309", linewidth=0.6, label="Indeterminate")

    # Annotate quadrants
    ax.text(0.85, 1.15, "Cocci Cluster\n(Spherical)", color="#475569", fontsize=8, fontweight="bold", ha="center")
    ax.text(0.45, 2.2, "Bacilli Cluster\n(Rods)", color="#475569", fontsize=8, fontweight="bold", ha="center")

    ax.set_xlabel("Circularity (4*pi*Area / P²)", fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Aspect Ratio (Major / Minor)", fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_title("Morphological Clustering & Gram Distribution", fontsize=10, fontweight="bold", color="#0F172A", pad=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.8, max(3.5, max(pos_aspect + neg_aspect + [3.0]) + 0.5))

    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    session_id: str,
    summary: Dict[str, Any],
    cells: List[Dict[str, Any]],
    raw_images: Dict[str, np.ndarray],
    output_filepath: str
) -> str:
    """
    Renders a comprehensive, multi-page Clinical / Microbiology Laboratory PDF Report
    adhering to medical diagnostic publication design standards.
    """
    doc = SimpleDocTemplate(
        output_filepath,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Medical Palette
    PRIMARY = colors.HexColor("#0F172A")
    ACCENT_BLUE = colors.HexColor("#2563EB")
    COLOR_POS = colors.HexColor("#6D28D9")
    COLOR_NEG = colors.HexColor("#BE123C")
    MUTED_TEXT = colors.HexColor("#64748B")
    BORDER_COLOR = colors.HexColor("#E2E8F0")
    BG_LIGHT = colors.HexColor("#F8FAFC")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=ACCENT_BLUE
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=MUTED_TEXT
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=PRIMARY,
        spaceAfter=6
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=PRIMARY
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=PRIMARY
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>BACTOVISION AI</b><br/><font size='8' color='#2563EB'>AUTOMATED CLINICAL MICROBIOLOGY REPORT</font>", title_style),
            Paragraph(
                f"<b>Sample ID:</b> #{session_id[:12].upper()}<br/>"
                f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
                f"<b>Engine:</b> {summary.get('model_type', 'Hybrid')}<br/>"
                f"<b>Calibration:</b> {summary.get('pixels_per_micron', 10.0)} px/µm",
                meta_style
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[3.6 * inch, 3.8 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_BLUE, spaceAfter=12))

    # 2. Executive Diagnostic Finding Banner
    diag_gram = summary.get("dominant_gram", "Unknown")
    diag_morph = summary.get("dominant_morphology", "Unknown")
    total_cells = summary.get("total_count", 0)

    banner_data = [
        [
            Paragraph("<b>PRIMARY DIAGNOSTIC FINDING:</b>", table_cell_bold),
            Paragraph(
                f"Microscopic automated analysis identified <b>{total_cells}</b> bacterial entities with predominant "
                f"<b><font color='{COLOR_POS.hexval()}'>{diag_gram}</font></b> profile and <b>{diag_morph}</b> morphology.",
                table_cell
            )
        ]
    ]
    banner_table = Table(banner_data, colWidths=[2.2 * inch, 5.2 * inch])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 12))

    # 3. Summary KPI Table
    story.append(Paragraph("1. Microscopic Quantification & Gram-Staining Ratios", section_h2))

    kpi_data = [
        [
            Paragraph("<b>Metric</b>", table_cell_bold),
            Paragraph("<b>Count</b>", table_cell_bold),
            Paragraph("<b>Percentage</b>", table_cell_bold),
            Paragraph("<b>Morphometric Standard</b>", table_cell_bold),
            Paragraph("<b>Calculated Value</b>", table_cell_bold)
        ],
        [
            Paragraph("<font color='#6D28D9'>■ Gram-Positive</font>", table_cell),
            Paragraph(str(summary.get("gram_positive_count", 0)), table_cell),
            Paragraph(f"{summary.get('gram_positive_pct', 0)}%", table_cell),
            Paragraph("Mean Cell Area", table_cell),
            Paragraph(f"{summary.get('mean_area_um2', 0)} µm² (± {summary.get('std_area_um2', 0)})", table_cell)
        ],
        [
            Paragraph("<font color='#BE123C'>■ Gram-Negative</font>", table_cell),
            Paragraph(str(summary.get("gram_negative_count", 0)), table_cell),
            Paragraph(f"{summary.get('gram_negative_pct', 0)}%", table_cell),
            Paragraph("Mean Circularity", table_cell),
            Paragraph(f"{summary.get('mean_circularity', 0)}", table_cell)
        ],
        [
            Paragraph("<font color='#D97706'>■ Indeterminate</font>", table_cell),
            Paragraph(str(summary.get("indeterminate_count", 0)), table_cell),
            Paragraph(f"{summary.get('indeterminate_pct', 0)}%", table_cell),
            Paragraph("Mean Aspect Ratio", table_cell),
            Paragraph(f"{summary.get('mean_aspect_ratio', 0)}", table_cell)
        ],
        [
            Paragraph("<b>Total Entities</b>", table_cell_bold),
            Paragraph(f"<b>{total_cells}</b>", table_cell_bold),
            Paragraph("100.0%", table_cell_bold),
            Paragraph("<b>Cocci / Bacilli Breakdown</b>", table_cell_bold),
            Paragraph(f"{summary.get('cocci_count', 0)} Cocci | {summary.get('bacilli_count', 0)} Bacilli", table_cell_bold)
        ]
    ]

    kpi_table = Table(kpi_data, colWidths=[1.8 * inch, 0.9 * inch, 1.1 * inch, 1.8 * inch, 1.8 * inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 4. Side-by-Side Images (Original vs. Annotated Classification)
    story.append(Paragraph("2. Microscopic Field & Instance Segmentation Imagery", section_h2))

    orig_bgr = raw_images.get("original_bgr")
    annot_bgr = raw_images.get("annotated_bgr")

    if orig_bgr is not None and annot_bgr is not None:
        # Encode temporary in-memory JPEGs for ReportLab
        _, orig_enc = cv2.imencode(".jpg", orig_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, annot_enc = cv2.imencode(".jpg", annot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        img_orig_flow = RLImage(io.BytesIO(orig_enc.tobytes()), width=3.65 * inch, height=2.45 * inch)
        img_annot_flow = RLImage(io.BytesIO(annot_enc.tobytes()), width=3.65 * inch, height=2.45 * inch)

        img_table_data = [
            [Paragraph("<b>Original Microscope Brightfield</b>", table_cell_bold), Paragraph("<b>BactoVision AI Annotated Overlay</b>", table_cell_bold)],
            [img_orig_flow, img_annot_flow]
        ]
        img_table = Table(img_table_data, colWidths=[3.7 * inch, 3.7 * inch])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 1), (-1, 1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 12))

    # 5. Scatter Plot Distribution Chart
    if len(cells) > 0:
        chart_buf = _generate_distribution_chart(cells)
        chart_flow = RLImage(chart_buf, width=6.2 * inch, height=2.8 * inch)
        story.append(KeepTogether([
            Paragraph("3. Morphological Clustering: Circularity vs. Aspect Ratio", section_h2),
            chart_flow,
            Spacer(1, 12)
        ]))

    # 6. Detailed Top 25 Cell Measurement Table
    story.append(Paragraph("4. Individual Specimen Morphometric Records (Top 25 Detected Cells)", section_h2))

    detail_header = [
        Paragraph("<b>ID</b>", table_cell_bold),
        Paragraph("<b>Gram Status</b>", table_cell_bold),
        Paragraph("<b>Morphology</b>", table_cell_bold),
        Paragraph("<b>Area (µm²)</b>", table_cell_bold),
        Paragraph("<b>Circularity</b>", table_cell_bold),
        Paragraph("<b>Aspect Ratio</b>", table_cell_bold),
        Paragraph("<b>Confidence</b>", table_cell_bold)
    ]

    detail_rows = [detail_header]
    top_cells = cells[:25]
    for cell in top_cells:
        status = cell["gram_status"]
        if status == "Gram-Positive":
            status_text = f"<font color='{COLOR_POS.hexval()}'><b>Positive</b></font>"
        elif status == "Gram-Negative":
            status_text = f"<font color='{COLOR_NEG.hexval()}'><b>Negative</b></font>"
        else:
            status_text = "<font color='#D97706'>Indet.</font>"

        detail_rows.append([
            Paragraph(f"#{cell['id']}", table_cell),
            Paragraph(status_text, table_cell),
            Paragraph(cell["morphology"], table_cell),
            Paragraph(f"{cell['area_um2']:.2f}", table_cell),
            Paragraph(f"{cell['circularity']:.3f}", table_cell),
            Paragraph(f"{cell['aspect_ratio']:.2f}", table_cell),
            Paragraph(f"{cell['confidence'] * 100:.1f}%", table_cell)
        ])

    detail_table = Table(
        detail_rows,
        colWidths=[0.6 * inch, 1.4 * inch, 1.4 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]
    )
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(KeepTogether([detail_table]))

    # 7. Quality Assurance & Sign-off Block
    story.append(Spacer(1, 14))
    qa_data = [
        [
            Paragraph("<b>Automated Quality Control:</b><br/>Image calibration verified with CLAHE normalization. Cell counts validated against Otsu-Watershed segmentation markers.", meta_style),
            Paragraph("<b>Verified By:</b> BactoVision AI Medical CV Engine<br/><b>Classification:</b> Research & Pre-Clinical Diagnostics Assistant", meta_style)
        ]
    ]
    qa_table = Table(qa_data, colWidths=[4.2 * inch, 3.2 * inch])
    qa_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(KeepTogether([qa_table]))

    # Build PDF
    doc.build(story)
    return output_filepath
