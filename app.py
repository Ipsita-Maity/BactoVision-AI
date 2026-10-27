"""
BactoVision AI - Automated Bacterial Gram-Staining Classification & Colony Counting Server
Flask RESTful API + Jinja2 frontend rendering + Export services.
"""

import os
import io
import uuid
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np

from cv_engine import BactoVisionPipeline
from report_generator import generate_csv, generate_pdf_report
from roboflow_client import RoboflowClient
from sample_generator import generate_synthetic_microscope_image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BactoVision.App")

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize Flask
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024 # 32MB max image upload
CORS(app)

# Pipeline & Roboflow singletons
cv_pipeline = BactoVisionPipeline()
roboflow_client = RoboflowClient()

# In-memory session store for export downloads (session_id -> {summary, cells, raw_images, timestamp})
session_store: Dict[str, Dict[str, Any]] = {}


@app.route("/", methods=["GET"])
def index():
    """Renders the main clinical dashboard UI."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint with system status."""
    return jsonify({
        "status": "healthy",
        "service": "BactoVision AI",
        "cv_pipeline": "active",
        "roboflow_connected": roboflow_client.is_connected
    })


@app.route("/api/generate-sample", methods=["POST"])
def generate_sample():
    """
    Generates a realistic synthetic Gram-stained microscopic slide or Petri dish image
    on-demand for zero-setup demo and testing.
    """
    try:
        data = request.get_json(silent=True) or {}
        sample_type = data.get("sample_type", "mixed")
        cell_count = int(data.get("cell_count", 65))
        is_circular = bool(data.get("is_circular", True))

        # Generate synthetic image
        img_bgr = generate_synthetic_microscope_image(
            width=1024,
            height=1024,
            sample_type=sample_type,
            num_cells=cell_count,
            is_circular=is_circular
        )

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        import base64
        b64_str = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "sample_type": sample_type,
            "filename": f"synthetic_{sample_type}_sample.jpg",
            "image_data": f"data:image/jpeg;base64,{b64_str}"
        })
    except Exception as e:
        logger.error(f"Sample generation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/roboflow/connect", methods=["POST"])
def roboflow_connect():
    """Configures Roboflow API credentials for hosted inference."""
    try:
        data = request.get_json() or {}
        api_key = data.get("api_key", "").strip()
        workspace = data.get("workspace", "").strip()
        project = data.get("project", "").strip()
        version = data.get("version", 1)

        result = roboflow_client.update_credentials(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Roboflow connect error: {e}")
        return jsonify({"success": False, "message": str(e)}), 400


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """
    Main Computer Vision analysis endpoint.
    Accepts multipart image and processing parameters, runs segmentation,
    Gram-staining HSV classification, and returns base64 images and metrics.
    """
    try:
        if "image" not in request.files:
            # Check if base64 image passed in JSON
            json_data = request.get_json(silent=True)
            if json_data and "image_b64" in json_data:
                import base64
                b64_val = json_data["image_b64"]
                if "," in b64_val:
                    b64_val = b64_val.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_val)
                form_data = json_data
            else:
                return jsonify({"error": "No image file provided in request."}), 400
        else:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "Empty filename provided."}), 400
            image_bytes = file.read()
            form_data = request.form

        # Extract parameters
        model_type = form_data.get("model_type", "watershed_hybrid")
        conf_threshold = float(form_data.get("conf_threshold", 0.35))
        pixels_per_micron = float(form_data.get("px_per_micron", 10.0))
        min_cell_area = int(form_data.get("min_cell_area", 25))
        max_cell_area = int(form_data.get("max_cell_area", 250000))

        # Optional HSV ranges fine-tuning
        hsv_params = {}
        for key in [
            "pos_h_min", "pos_h_max", "pos_s_min", "pos_v_min",
            "neg_h1_min", "neg_h1_max", "neg_h2_min", "neg_h2_max",
            "neg_s_min", "neg_v_min"
        ]:
            if key in form_data:
                try:
                    hsv_params[key] = int(form_data[key])
                except (ValueError, TypeError):
                    pass

        # Execute CV Pipeline
        result = cv_pipeline.analyze(
            image_bytes=image_bytes,
            model_type=model_type,
            conf_threshold=conf_threshold,
            pixels_per_micron=pixels_per_micron,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
            hsv_params=hsv_params if hsv_params else None,
            roboflow_client=roboflow_client if model_type == "roboflow_api" else None
        )

        # Generate unique session ID for reporting exports
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "summary": result["summary"],
            "cells": result["cells"],
            "raw_images": result["_raw_images"]
        }

        # Clean old sessions if cache grows large
        if len(session_store) > 100:
            oldest_key = next(iter(session_store))
            session_store.pop(oldest_key, None)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "summary": result["summary"],
            "cells": result["cells"],
            "images": result["images"]
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export/csv/<session_id>", methods=["GET"])
def export_csv(session_id: str):
    """Generates and downloads a detailed CSV file of all cell morphometric measurements."""
    session = session_store.get(session_id)
    if not session:
        return jsonify({"error": "Session expired or invalid session ID."}), 404

    csv_content = generate_csv(session["summary"], session["cells"])
    bio = io.BytesIO(csv_content.encode("utf-8"))
    bio.seek(0)

    filename = f"bactovision_analysis_{session_id[:8]}.csv"
    return send_file(
        bio,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/export/pdf/<session_id>", methods=["GET"])
def export_pdf(session_id: str):
    """Generates and downloads a medical-grade Clinical Microbiology Lab PDF Report."""
    session = session_store.get(session_id)
    if not session:
        return jsonify({"error": "Session expired or invalid session ID."}), 404

    pdf_filename = f"bactovision_lab_report_{session_id[:8]}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    try:
        generate_pdf_report(
            session_id=session_id,
            summary=session["summary"],
            cells=session["cells"],
            raw_images=session["raw_images"],
            output_filepath=pdf_path
        )

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_filename
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting BactoVision AI server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
