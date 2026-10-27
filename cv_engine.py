"""
BactoVision AI - Computer Vision & Deep Learning Engine
Performs CLAHE illumination normalization, hybrid deep learning / watershed instance segmentation,
Gram-staining HSV feature extraction, morphometric measurement, and visual annotation.
"""

import os
import io
import math
import base64
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger("BactoVision.CVEngine")

# Optional deep learning backends with graceful fallbacks
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.info("Ultralytics YOLO not installed in current environment; classical CV fallback active.")

TORCH_AVAILABLE = False
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    logger.info("PyTorch / Torchvision not available; classical CV fallback active.")


class BactoVisionPipeline:
    """
    Production-grade Computer Vision pipeline for microscopic bacterial Gram-staining
    and colony morphology analysis.
    """

    def __init__(self, yolo_model_path: str = "yolov8n-seg.pt"):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None

    def _get_yolo_model(self):
        """Lazy-loads YOLOv8-seg model if ultralytics is available."""
        if not YOLO_AVAILABLE:
            return None
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO(self.yolo_model_path)
            except Exception as e:
                logger.warning(f"Could not load YOLOv8 model from {self.yolo_model_path}: {e}")
                self._yolo_model = None
        return self._yolo_model

    # -------------------------------------------------------------------------
    # 1. IMAGE PREPROCESSING
    # -------------------------------------------------------------------------
    def preprocess_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Applies noise filtering and CLAHE (Contrast Limited Adaptive Histogram Equalization)
        in the CIE LAB color space to equalize uneven microscope condenser illumination.
        """
        # Step 1: Subtle bilateral filter to preserve cell membrane edges while reducing sensor grain
        denoised = cv2.bilateralFilter(bgr_image, d=7, sigmaColor=45, sigmaSpace=45)

        # Step 2: Convert to LAB color space
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Step 3: Apply CLAHE to the Luminance (L) channel
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        cl_channel = clahe.apply(l_channel)

        # Step 4: Merge back and return BGR
        enhanced_lab = cv2.merge([cl_channel, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    # -------------------------------------------------------------------------
    # 2. SEGMENTATION PIPELINE (WATERSHED & DEEP LEARNING)
    # -------------------------------------------------------------------------
    def segment_cells_watershed(
        self,
        bgr_image: np.ndarray,
        min_area: int = 25,
        max_area: int = 250000
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Classical CV: Adaptive/Otsu Thresholding + Morphological Opening + Distance Transform + Watershed.
        Accurately separates overlapping bacterial cells and clusters.
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Detect whether the image has a bright brightfield background or darkfield
        mean_brightness = np.mean(gray)
        is_brightfield = mean_brightness > 115

        if is_brightfield:
            # Bacteria are darker than background
            # Invert grayscale for distance transform calculations
            inv_gray = cv2.bitwise_not(gray)
            # Use Otsu's thresholding
            _, binary = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Mask out dark optical surround if circular aperture exists
            corner_mean = (float(gray[0, 0]) + float(gray[0, -1]) + float(gray[-1, 0]) + float(gray[-1, -1])) / 4.0
            if corner_mean < 45:
                # Circular field mask
                field_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(field_mask, (w // 2, h // 2), int(min(h, w) * 0.46), 255, -1)
                binary = cv2.bitwise_and(binary, binary, mask=field_mask)
        else:
            # Darkfield / Agar plate with bright colonies
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological noise removal (Opening)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Sure background area via dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area using Distance Transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            _, sure_fg = cv2.threshold(dist_transform, 0.28 * max_dist, 255, 0)
        else:
            sure_fg = opening.copy()
        sure_fg = np.uint8(sure_fg)

        # Unknown region = sure_bg - sure_fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling for Watershed
        num_markers, markers = cv2.connectedComponents(sure_fg)
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Mark unknown region as 0
        markers[unknown == 255] = 0

        # Apply Watershed algorithm
        markers_img = bgr_image.copy()
        markers = cv2.watershed(markers_img, markers)

        # Extract contours from watershed markers
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        contours_list = []

        # In watershed, boundaries are marked -1, background is 1, individual cells are >= 2
        separated_fg = np.zeros((h, w), dtype=np.uint8)
        separated_fg[markers > 1] = 255
        separated_fg[markers <= 1] = 0

        cnts, _ = cv2.findContours(separated_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                contours_list.append(cnt)
                cv2.drawContours(binary_mask, [cnt], -1, 255, -1)

        return contours_list, binary_mask

    def segment_cells_yolo(
        self,
        bgr_image: np.ndarray,
        conf_threshold: float = 0.35,
        min_area: int = 25,
        max_area: int = 250000
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Deep Learning Mode: Uses YOLOv8 segmentation model to predict instance masks.
        Falls back to Watershed if YOLO model is unavailable or weights are not yet downloaded.
        """
        model = self._get_yolo_model()
        if model is None:
            logger.info("YOLOv8 unavailable. Running Watershed segmentation fallback.")
            return self.segment_cells_watershed(bgr_image, min_area, max_area)

        try:
            h, w = bgr_image.shape[:2]
            results = model.predict(source=bgr_image, conf=conf_threshold, verbose=False)
            contours_list = []
            binary_mask = np.zeros((h, w), dtype=np.uint8)

            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy() # [N, H, W]
                for i in range(len(masks)):
                    mask_i = (masks[i] * 255).astype(np.uint8)
                    if mask_i.shape[:2] != (h, w):
                        mask_i = cv2.resize(mask_i, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        area = cv2.contourArea(cnt)
                        if min_area <= area <= max_area:
                            contours_list.append(cnt)
                            cv2.drawContours(binary_mask, [cnt], -1, 255, -1)

            if len(contours_list) == 0:
                # If YOLO model returned 0 detections on custom microscope slide, fallback to watershed
                logger.info("YOLO found 0 detections on microscope image. Engaging Watershed fallback.")
                return self.segment_cells_watershed(bgr_image, min_area, max_area)

            return contours_list, binary_mask
        except Exception as e:
            logger.error(f"YOLO segmentation failed: {e}. Falling back to Watershed.")
            return self.segment_cells_watershed(bgr_image, min_area, max_area)

    # -------------------------------------------------------------------------
    # 3. GRAM-STAINING HSV CLASSIFICATION
    # -------------------------------------------------------------------------
    def classify_gram_status(
        self,
        bgr_image: np.ndarray,
        contour: np.ndarray,
        hsv_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies segmented cell ROI using HSV color thresholding into:
        - Gram-Positive (Crystal Violet / Deep Indigo / Purple)
        - Gram-Negative (Safranin / Magenta / Red-Pink)
        - Indeterminate
        """
        # Default calibrated HSV ranges
        params = hsv_params or {}
        pos_h_min = params.get("pos_h_min", 115)
        pos_h_max = params.get("pos_h_max", 165)
        pos_s_min = params.get("pos_s_min", 40)
        pos_v_min = params.get("pos_v_min", 30)

        neg_h1_min = params.get("neg_h1_min", 0)
        neg_h1_max = params.get("neg_h1_max", 15)
        neg_h2_min = params.get("neg_h2_min", 166)
        neg_h2_max = params.get("neg_h2_max", 180)
        neg_s_min = params.get("neg_s_min", 30)
        neg_v_min = params.get("neg_v_min", 50)

        # Create ROI mask for this specific contour
        x, y, w, h = cv2.boundingRect(contour)
        h_img, w_img = bgr_image.shape[:2]
        
        # Guard bounding box within image boundaries
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        roi_bgr = bgr_image[y1:y2, x1:x2]
        if roi_bgr.size == 0 or w <= 0 or h <= 0:
            return "Indeterminate", 0.0, {"h": 0.0, "s": 0.0, "v": 0.0}

        roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        offset_cnt = contour - [x1, y1]
        cv2.drawContours(roi_mask, [offset_cnt], -1, 255, -1)

        total_mask_pixels = np.count_nonzero(roi_mask)
        if total_mask_pixels == 0:
            return "Indeterminate", 0.0, {"h": 0.0, "s": 0.0, "v": 0.0}

        # Convert ROI to HSV
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Calculate mean HSV values inside mask
        mean_hsv = cv2.mean(roi_hsv, mask=roi_mask)
        mean_h, mean_s, mean_v = mean_hsv[0], mean_hsv[1], mean_hsv[2]

        # 1. Gram-Positive Mask: Purple / Indigo / Crystal Violet
        lower_pos = np.array([pos_h_min, pos_s_min, pos_v_min], dtype=np.uint8)
        upper_pos = np.array([pos_h_max, 255, 255], dtype=np.uint8)
        pos_mask = cv2.inRange(roi_hsv, lower_pos, upper_pos)
        pos_mask = cv2.bitwise_and(pos_mask, pos_mask, mask=roi_mask)
        pos_pixels = np.count_nonzero(pos_mask)

        # 2. Gram-Negative Mask: Dual Safranin Pink/Red range
        lower_neg1 = np.array([neg_h1_min, neg_s_min, neg_v_min], dtype=np.uint8)
        upper_neg1 = np.array([neg_h1_max, 255, 255], dtype=np.uint8)
        mask_neg1 = cv2.inRange(roi_hsv, lower_neg1, upper_neg1)

        lower_neg2 = np.array([neg_h2_min, neg_s_min, neg_v_min], dtype=np.uint8)
        upper_neg2 = np.array([neg_h2_max, 255, 255], dtype=np.uint8)
        mask_neg2 = cv2.inRange(roi_hsv, lower_neg2, upper_neg2)

        neg_mask = cv2.bitwise_or(mask_neg1, mask_neg2)
        neg_mask = cv2.bitwise_and(neg_mask, neg_mask, mask=roi_mask)
        neg_pixels = np.count_nonzero(neg_mask)

        # Ratios
        pos_ratio = pos_pixels / float(total_mask_pixels)
        neg_ratio = neg_pixels / float(total_mask_pixels)

        # Color Classification Decision Logic
        # Also use mean Hue if pixel ratio is near parity
        is_hue_pos = (pos_h_min <= mean_h <= pos_h_max) and (mean_s >= 25)
        is_hue_neg = ((neg_h1_min <= mean_h <= neg_h1_max) or (neg_h2_min <= mean_h <= neg_h2_max)) and (mean_s >= 25)

        if pos_ratio > neg_ratio and (pos_ratio >= 0.12 or is_hue_pos):
            status = "Gram-Positive"
            conf = min(0.99, 0.50 + pos_ratio * 0.50)
        elif neg_ratio > pos_ratio and (neg_ratio >= 0.12 or is_hue_neg):
            status = "Gram-Negative"
            conf = min(0.99, 0.50 + neg_ratio * 0.50)
        elif is_hue_pos and not is_hue_neg:
            status = "Gram-Positive"
            conf = 0.65
        elif is_hue_neg and not is_hue_pos:
            status = "Gram-Negative"
            conf = 0.65
        else:
            status = "Indeterminate"
            conf = 0.35

        return status, round(conf, 3), {
            "h": round(mean_h, 1),
            "s": round(mean_s, 1),
            "v": round(mean_v, 1)
        }

    # -------------------------------------------------------------------------
    # 4. MORPHOMETRICS & MORPHOLOGY CLASSIFICATION
    # -------------------------------------------------------------------------
    def compute_morphometrics(
        self,
        contour: np.ndarray,
        pixels_per_micron: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calculates geometric and morphometric parameters:
        - Area (px² and µm²)
        - Perimeter (px and µm)
        - Circularity (4 * pi * Area / Perimeter²)
        - Aspect Ratio (Major / Minor axis)
        - Morphology: 'Cocci' (spherical), 'Bacilli' (rod-shaped), or 'Coccobacilli / Other'
        """
        area_px = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, True))

        # Calibration to microns
        scale = max(pixels_per_micron, 0.001)
        area_um2 = area_px / (scale ** 2)
        perimeter_um = perimeter_px / scale

        # Circularity formula: 4 * pi * Area / Perimeter^2
        if perimeter_px > 0:
            circularity = (4.0 * math.pi * area_px) / (perimeter_px ** 2)
            circularity = max(0.0, min(1.0, circularity))
        else:
            circularity = 0.0

        # Aspect Ratio via minimum area rectangle or ellipse fit
        aspect_ratio = 1.0
        major_axis = 1.0
        minor_axis = 1.0

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (d1, d2), angle = ellipse
                major_axis = max(d1, d2)
                minor_axis = max(min(d1, d2), 1.0)
                aspect_ratio = major_axis / minor_axis
            except Exception:
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rw, rh), angle = rect
                major_axis = max(rw, rh)
                minor_axis = max(min(rw, rh), 1.0)
                aspect_ratio = major_axis / minor_axis
        else:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            major_axis = max(rw, rh)
            minor_axis = max(min(rw, rh), 1.0)
            aspect_ratio = major_axis / minor_axis

        # Morphology Classification Rule
        # Cocci: Circularity >= 0.78 and Aspect Ratio < 1.35
        # Bacilli: Aspect Ratio >= 1.35 and Circularity < 0.78
        # Coccobacilli / Other: intermediate
        if circularity >= 0.78 and aspect_ratio < 1.35:
            morphology = "Cocci"
        elif aspect_ratio >= 1.35 and circularity < 0.78:
            morphology = "Bacilli"
        else:
            morphology = "Coccobacilli / Other"

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x = int(np.mean(contour[:, 0, 0]))
            centroid_y = int(np.mean(contour[:, 0, 1]))

        x, y, w, h = cv2.boundingRect(contour)

        return {
            "area_px": round(area_px, 1),
            "area_um2": round(area_um2, 2),
            "perimeter_px": round(perimeter_px, 1),
            "perimeter_um": round(perimeter_um, 2),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "morphology": morphology,
            "centroid": [centroid_x, centroid_y],
            "bbox": [int(x), int(y), int(w), int(h)]
        }

    # -------------------------------------------------------------------------
    # 5. VISUAL ANNOTATION & SCALE BAR
    # -------------------------------------------------------------------------
    def annotate_image(
        self,
        bgr_image: np.ndarray,
        cells: List[Dict[str, Any]],
        contours: List[np.ndarray],
        pixels_per_micron: float = 10.0
    ) -> np.ndarray:
        """
        Generates clinical annotation visualization with color-coded contours,
        alpha-blended masks, cell IDs, morphology tags, and calibrated scale bar.
        """
        annotated = bgr_image.copy()
        overlay = annotated.copy()
        h, w = annotated.shape[:2]

        # Colors in BGR
        # Gram-Positive: Rich Violet / Crystal Violet Indigo
        COLOR_POS = (195, 45, 125)
        # Gram-Negative: Vibrant Safranin Pink-Red
        COLOR_NEG = (85, 45, 235)
        # Indeterminate: Cyan / Amber
        COLOR_INDET = (40, 190, 240)

        for cell, cnt in zip(cells, contours):
            status = cell["gram_status"]
            if status == "Gram-Positive":
                color = COLOR_POS
            elif status == "Gram-Negative":
                color = COLOR_NEG
            else:
                color = COLOR_INDET

            # Fill overlay mask
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            # Crisp contour boundary on main image
            cv2.drawContours(annotated, [cnt], -1, color, 2)

        # Alpha blend masks for translucent fluorescent effect
        annotated = cv2.addWeighted(overlay, 0.32, annotated, 0.68, 0)

        # Draw Labels & Centroids
        for cell in cells:
            cid = cell["id"]
            cx, cy = cell["centroid"]
            morph = cell["morphology"]
            status = cell["gram_status"]

            # Centroid point
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

            # Draw concise label pill for top cells or all if count < 120
            if len(cells) <= 120 or cid <= 60:
                short_morph = "C" if morph == "Cocci" else ("B" if morph == "Bacilli" else "O")
                label_text = f"#{cid} {short_morph}"
                
                # Font metrics
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.38
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                lx = min(w - tw - 4, max(4, cx + 5))
                ly = max(th + 4, cy - 5)

                # Background pill
                pill_bg = (18, 20, 24)
                cv2.rectangle(annotated, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), pill_bg, -1)
                
                # Pill border color
                border_color = COLOR_POS if status == "Gram-Positive" else (COLOR_NEG if status == "Gram-Negative" else COLOR_INDET)
                cv2.rectangle(annotated, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), border_color, 1)

                cv2.putText(annotated, label_text, (lx, ly), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw Calibrated Scale Bar (e.g. 10 µm or 25 µm)
        scale = max(pixels_per_micron, 0.001)
        # Choose suitable bar length: 10 µm or 25 µm or 50 µm
        target_um = 10.0 if (scale * 10.0) >= 40 else (25.0 if (scale * 25.0) >= 40 else 50.0)
        bar_len_px = int(target_um * scale)

        # Bottom right placement
        margin_x = 35
        margin_y = 35
        bar_x2 = w - margin_x
        bar_x1 = bar_x2 - bar_len_px
        bar_y = h - margin_y

        if bar_x1 > 10 and bar_y > 30:
            # Scale bar container pill
            box_pad = 10
            cv2.rectangle(
                annotated,
                (bar_x1 - box_pad, bar_y - 24),
                (bar_x2 + box_pad, bar_y + 12),
                (20, 22, 26),
                -1
            )
            cv2.rectangle(
                annotated,
                (bar_x1 - box_pad, bar_y - 24),
                (bar_x2 + box_pad, bar_y + 12),
                (60, 65, 75),
                1
            )
            # Scale bar line
            cv2.line(annotated, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), 3)
            # Text label
            label = f"{int(target_um)} um"
            cv2.putText(annotated, label, (bar_x1 + 4, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        return annotated

    # -------------------------------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # -------------------------------------------------------------------------
    def analyze(
        self,
        image_bytes: bytes,
        model_type: str = "watershed_hybrid",
        conf_threshold: float = 0.35,
        pixels_per_micron: float = 10.0,
        min_cell_area: int = 25,
        max_cell_area: int = 250000,
        hsv_params: Optional[Dict[str, Any]] = None,
        roboflow_client = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on raw image bytes.
        Returns base64 encoded views and complete per-cell data + summary statistics.
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise ValueError("Invalid image format or corrupted image buffer.")

        # Stage 1: CLAHE & Illumination Preprocessing
        preprocessed_bgr = self.preprocess_image(original_bgr)

        # Stage 2: Segmentation
        contours: List[np.ndarray] = []
        binary_mask: np.ndarray

        if model_type == "roboflow_api" and roboflow_client and roboflow_client.is_connected:
            try:
                rf_res = roboflow_client.infer_image(preprocessed_bgr, confidence=conf_threshold)
                if rf_res.get("success"):
                    # Process Roboflow predictions into contours
                    preds = rf_res.get("predictions", [])
                    h_img, w_img = preprocessed_bgr.shape[:2]
                    binary_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    for pred in preds:
                        if "points" in pred: # segmentation polygon
                            pts = np.array([[int(p["x"]), int(p["y"])] for p in pred["points"]], dtype=np.int32)
                            contours.append(pts)
                            cv2.drawContours(binary_mask, [pts], -1, 255, -1)
                        else: # bounding box
                            bx = int(pred["x"] - pred["width"] / 2)
                            by = int(pred["y"] - pred["height"] / 2)
                            bw = int(pred["width"])
                            bh = int(pred["height"])
                            cnt = np.array([[[bx, by]], [[bx+bw, by]], [[bx+bw, by+bh]], [[bx, by+bh]]], dtype=np.int32)
                            contours.append(cnt)
                            cv2.drawContours(binary_mask, [cnt], -1, 255, -1)
                else:
                    logger.warning(f"Roboflow inference failed: {rf_res.get('error')}. Using Watershed fallback.")
                    contours, binary_mask = self.segment_cells_watershed(preprocessed_bgr, min_cell_area, max_cell_area)
            except Exception as e:
                logger.error(f"Roboflow inference exception: {e}. Fallback to Watershed.")
                contours, binary_mask = self.segment_cells_watershed(preprocessed_bgr, min_cell_area, max_cell_area)
        elif model_type in ["yolov8", "yolov8_seg"]:
            contours, binary_mask = self.segment_cells_yolo(preprocessed_bgr, conf_threshold, min_cell_area, max_cell_area)
        else: # watershed_hybrid or fallback
            contours, binary_mask = self.segment_cells_watershed(preprocessed_bgr, min_cell_area, max_cell_area)

        # Stage 3 & 4: Per-Cell Feature Extraction, Gram Classification, Morphometrics
        cells: List[Dict[str, Any]] = []
        valid_contours = []

        for idx, cnt in enumerate(contours, start=1):
            morph = self.compute_morphometrics(cnt, pixels_per_micron)
            gram_status, conf, mean_hsv = self.classify_gram_status(preprocessed_bgr, cnt, hsv_params)

            cell_obj = {
                "id": idx,
                "gram_status": gram_status,
                "confidence": conf,
                "morphology": morph["morphology"],
                "area_px": morph["area_px"],
                "area_um2": morph["area_um2"],
                "perimeter_px": morph["perimeter_px"],
                "perimeter_um": morph["perimeter_um"],
                "circularity": morph["circularity"],
                "aspect_ratio": morph["aspect_ratio"],
                "mean_hsv": mean_hsv,
                "centroid": morph["centroid"],
                "bbox": morph["bbox"]
            }
            cells.append(cell_obj)
            valid_contours.append(cnt)

        # Stage 5: Visual Annotation
        annotated_bgr = self.annotate_image(preprocessed_bgr, cells, valid_contours, pixels_per_micron)

        # Compute Summary Statistics
        total_count = len(cells)
        pos_count = sum(1 for c in cells if c["gram_status"] == "Gram-Positive")
        neg_count = sum(1 for c in cells if c["gram_status"] == "Gram-Negative")
        indet_count = sum(1 for c in cells if c["gram_status"] == "Indeterminate")

        cocci_count = sum(1 for c in cells if c["morphology"] == "Cocci")
        bacilli_count = sum(1 for c in cells if c["morphology"] == "Bacilli")
        other_count = total_count - (cocci_count + bacilli_count)

        pos_pct = round((pos_count / total_count * 100.0) if total_count > 0 else 0.0, 1)
        neg_pct = round((neg_count / total_count * 100.0) if total_count > 0 else 0.0, 1)
        indet_pct = round((indet_count / total_count * 100.0) if total_count > 0 else 0.0, 1)

        areas_um2 = [c["area_um2"] for c in cells]
        circs = [c["circularity"] for c in cells]
        aspects = [c["aspect_ratio"] for c in cells]

        mean_area_um2 = round(float(np.mean(areas_um2)), 2) if areas_um2 else 0.0
        std_area_um2 = round(float(np.std(areas_um2)), 2) if areas_um2 else 0.0
        mean_circularity = round(float(np.mean(circs)), 3) if circs else 0.0
        mean_aspect_ratio = round(float(np.mean(aspects)), 2) if aspects else 0.0

        if pos_count > neg_count:
            dominant_gram = "Gram-Positive"
        elif neg_count > pos_count:
            dominant_gram = "Gram-Negative"
        else:
            dominant_gram = "Mixed / Indeterminate"

        if cocci_count >= bacilli_count and cocci_count >= other_count:
            dominant_morph = "Cocci"
        elif bacilli_count >= cocci_count and bacilli_count >= other_count:
            dominant_morph = "Bacilli"
        else:
            dominant_morph = "Mixed Morphology"

        summary = {
            "total_count": total_count,
            "gram_positive_count": pos_count,
            "gram_positive_pct": pos_pct,
            "gram_negative_count": neg_count,
            "gram_negative_pct": neg_pct,
            "indeterminate_count": indet_count,
            "indeterminate_pct": indet_pct,
            "cocci_count": cocci_count,
            "bacilli_count": bacilli_count,
            "other_count": other_count,
            "mean_area_um2": mean_area_um2,
            "std_area_um2": std_area_um2,
            "mean_circularity": mean_circularity,
            "mean_aspect_ratio": mean_aspect_ratio,
            "dominant_gram": dominant_gram,
            "dominant_morphology": dominant_morph,
            "pixels_per_micron": pixels_per_micron,
            "model_type": model_type
        }

        # Convert images to Base64 strings for REST API response
        def to_b64(img: np.ndarray, is_gray: bool = False) -> str:
            if is_gray:
                _, buf = cv2.imencode(".png", img)
            else:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buf).decode("utf-8")

        return {
            "summary": summary,
            "cells": cells,
            "images": {
                "original": f"data:image/jpeg;base64,{to_b64(original_bgr)}",
                "preprocessed": f"data:image/jpeg;base64,{to_b64(preprocessed_bgr)}",
                "mask": f"data:image/png;base64,{to_b64(binary_mask, is_gray=True)}",
                "annotated": f"data:image/jpeg;base64,{to_b64(annotated_bgr)}"
            },
            # Raw numpy image buffers for report generator
            "_raw_images": {
                "original_bgr": original_bgr,
                "preprocessed_bgr": preprocessed_bgr,
                "mask_gray": binary_mask,
                "annotated_bgr": annotated_bgr
            }
        }
