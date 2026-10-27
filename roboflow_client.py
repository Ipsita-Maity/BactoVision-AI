"""
BactoVision AI - Roboflow Integration Client
Supports cloud-hosted inference, dataset retrieval, and custom model loading
via the Roboflow Python SDK and REST API.
"""

import os
import json
import logging
import base64
import requests
import numpy as np
import cv2

logger = logging.getLogger("BactoVision.Roboflow")


class RoboflowClient:
    """Wrapper for Roboflow API connections and remote inference."""

    def __init__(self, api_key: str = None, workspace: str = None, project: str = None, version: int = 1):
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self.workspace = workspace or os.environ.get("ROBOFLOW_WORKSPACE", "")
        self.project = project or os.environ.get("ROBOFLOW_PROJECT", "")
        self.version = int(version or os.environ.get("ROBOFLOW_VERSION", 1))
        self.is_connected = False
        if self.api_key and self.project:
            self.test_connection()

    def update_credentials(self, api_key: str, workspace: str = "", project: str = "", version: int = 1) -> dict:
        """Updates and validates Roboflow credentials."""
        self.api_key = api_key.strip()
        self.workspace = workspace.strip()
        self.project = project.strip()
        try:
            self.version = int(version)
        except (ValueError, TypeError):
            self.version = 1
            
        return self.test_connection()

    def test_connection(self) -> dict:
        """Verifies API key and project access against the Roboflow API."""
        if not self.api_key:
            self.is_connected = False
            return {
                "success": False,
                "message": "Roboflow API key is required."
            }

        url = f"https://api.roboflow.com/?api_key={self.api_key}"
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                workspace_list = list(data.get("workspaces", {}).keys())
                self.is_connected = True
                return {
                    "success": True,
                    "message": "Successfully authenticated with Roboflow API.",
                    "workspaces": workspace_list,
                    "active_project": self.project,
                    "active_version": self.version
                }
            else:
                self.is_connected = False
                return {
                    "success": False,
                    "message": f"Roboflow API error: HTTP {resp.status_code} - {resp.text}"
                }
        except Exception as e:
            logger.error(f"Failed to connect to Roboflow: {e}")
            self.is_connected = False
            return {
                "success": False,
                "message": f"Network error connecting to Roboflow: {str(e)}"
            }

    def infer_image(self, image_bgr: np.ndarray, confidence: float = 0.40) -> dict:
        """
        Runs remote hosted inference on an uploaded image via Roboflow Serverless Inference API.
        Returns parsed bounding boxes and segmentation polygons if model is instance segmentation.
        """
        if not self.api_key or not self.project:
            raise ValueError("Roboflow API Key and Project ID must be configured before running inference.")

        # Encode image to JPEG base64
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Hosted inference endpoint
        # Format: https://detect.roboflow.com/[PROJECT]/[VERSION]?api_key=[API_KEY]&confidence=[CONF]
        endpoint = (
            f"https://detect.roboflow.com/{self.project}/{self.version}"
            f"?api_key={self.api_key}&confidence={confidence}&format=json"
        )

        try:
            resp = requests.post(
                endpoint,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=25
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Roboflow inference returned HTTP {resp.status_code}: {resp.text}",
                    "predictions": []
                }
            
            result = resp.json()
            predictions = result.get("predictions", [])
            return {
                "success": True,
                "predictions": predictions,
                "image_width": result.get("image", {}).get("width", image_bgr.shape[1]),
                "image_height": result.get("image", {}).get("height", image_bgr.shape[0])
            }
        except Exception as e:
            logger.error(f"Roboflow inference exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": []
            }
