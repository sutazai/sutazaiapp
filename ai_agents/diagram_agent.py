"""
Diagram Agent Module

This module provides a specialized agent for parsing and analyzing diagrams.
It includes capabilities for detecting shapes, text, and relationships.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract

from .base_agent import BaseAgent, AgentError

logger = logging.getLogger(__name__)


class DiagramAgent(BaseAgent):
    """Agent specialized for diagram parsing and analysis."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the diagram agent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.supported_formats = ["png", "jpg", "jpeg", "svg"]
        self.min_shape_area = config.get("min_shape_area", 100)
        self.max_shape_area = config.get("max_shape_area", 10000)
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are available.

        Raises:
            AgentError: If dependencies are missing
        """
        try:
            # Check OpenCV
            cv2.__version__
        except Exception as e:
            logger.warning(f"OpenCV not available: {str(e)}")
            self.opencv_available = False
        else:
            self.opencv_available = True

        try:
            # Check Tesseract
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
            self.tesseract_available = False
        else:
            self.tesseract_available = True

    def _initialize(self) -> None:
        """Initialize the diagram agent."""
        # Create temporary directory for processing
        self.temp_dir = Path("/tmp/diagram_agent")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenCV if available
        if self.opencv_available:
            logger.info("OpenCV initialized")

        # Initialize OCR if available
        if self.tesseract_available:
            logger.info("Tesseract OCR initialized")

    def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a diagram processing task.

        Args:
            task: Task configuration dictionary

        Returns:
            Dict[str, Any]: Task results

        Raises:
            AgentError: If task execution fails
        """
        task_type = task["type"]
        params = task["parameters"]

        if task_type == "detect_shapes":
            return self._detect_shapes(params)
        elif task_type == "extract_text":
            return self._extract_text(params)
        elif task_type == "analyze_diagram":
            return self._analyze_diagram(params)
        else:
            raise AgentError(f"Unsupported task type: {task_type}")

    def _cleanup(self) -> None:
        """Clean up diagram agent resources."""
        # Remove temporary files
        try:
            for file in self.temp_dir.glob("*"):
                file.unlink()
            self.temp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def _detect_shapes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect shapes in a diagram.

        Args:
            params: Task parameters

        Returns:
            Dict[str, Any]: Detected shapes and metadata

        Raises:
            AgentError: If shape detection fails
        """
        if not self.opencv_available:
            raise AgentError("OpenCV not available")

        file_path = params.get("file_path")
        if not file_path:
            raise AgentError("Missing file_path parameter")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")

            # Get file extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.supported_formats:
                raise AgentError(f"Unsupported file format: {ext}")

            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                raise AgentError(f"Failed to read image: {file_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process contours
            shapes = []
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                if area < self.min_shape_area or area > self.max_shape_area:
                    continue

                # Get shape type
                shape_type = self._get_shape_type(contour)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Get center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                shapes.append(
                    {
                        "type": shape_type,
                        "area": area,
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": cx,
                            "center_y": cy,
                        },
                    }
                )

            return {
                "file_path": str(file_path),
                "file_type": ext,
                "shape_count": len(shapes),
                "shapes": shapes,
            }

        except Exception as e:
            logger.error(f"Error detecting shapes: {str(e)}")
            raise AgentError(f"Failed to detect shapes: {str(e)}")

    def _get_shape_type(self, contour: np.ndarray) -> str:
        """
        Determine the type of shape from its contour.

        Args:
            contour: OpenCV contour array

        Returns:
            str: Shape type identifier
        """
        # Get perimeter
        perimeter = cv2.arcLength(contour, True)

        # Approximate shape
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Count vertices
        vertices = len(approx)

        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices == 6:
            return "hexagon"
        elif vertices > 6:
            return "circle"
        else:
            return "unknown"

    def _extract_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from a diagram.

        Args:
            params: Task parameters

        Returns:
            Dict[str, Any]: Extracted text and metadata

        Raises:
            AgentError: If text extraction fails
        """
        if not self.tesseract_available:
            raise AgentError("OCR not available")

        file_path = params.get("file_path")
        if not file_path:
            raise AgentError("Missing file_path parameter")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")

            # Get file extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.supported_formats:
                raise AgentError(f"Unsupported file format: {ext}")

            # Open image
            image = Image.open(file_path)

            # Extract text
            text = pytesseract.image_to_string(image)

            return {
                "file_path": str(file_path),
                "file_type": ext,
                "text": text,
                "word_count": len(text.split()),
                "line_count": len(text.splitlines()),
            }

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise AgentError(f"Failed to extract text: {str(e)}")

    def _analyze_diagram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a diagram for various features.

        Args:
            params: Task parameters

        Returns:
            Dict[str, Any]: Analysis results

        Raises:
            AgentError: If analysis fails
        """
        file_path = params.get("file_path")
        if not file_path:
            raise AgentError("Missing file_path parameter")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")

            # Get file extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.supported_formats:
                raise AgentError(f"Unsupported file format: {ext}")

            # Detect shapes
            shapes_result = self._detect_shapes({"file_path": str(file_path)})

            # Extract text
            text_result = self._extract_text({"file_path": str(file_path)})

            # Perform analysis
            analysis = {
                "file_path": str(file_path),
                "file_type": ext,
                "file_size": file_path.stat().st_size,
                "shape_count": shapes_result["shape_count"],
                "text_count": text_result["word_count"],
                "line_count": text_result["line_count"],
                "shapes": shapes_result["shapes"],
                "text": text_result["text"],
            }

            # Calculate additional metrics
            if shapes_result["shapes"]:
                areas = [shape["area"] for shape in shapes_result["shapes"]]
                analysis.update(
                    {
                        "average_shape_area": np.mean(areas),
                        "max_shape_area": max(areas),
                        "min_shape_area": min(areas),
                        "shape_types": {
                            shape["type"]: len(
                                [
                                    s
                                    for s in shapes_result["shapes"]
                                    if s["type"] == shape["type"]
                                ]
                            )
                            for shape in shapes_result["shapes"]
                        },
                    }
                )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing diagram: {str(e)}")
            raise AgentError(f"Failed to analyze diagram: {str(e)}")

    def _update_metrics(self, execution_time: float) -> None:
        """
        Update agent metrics.

        Args:
            execution_time: Task execution time in seconds
        """
        # Add custom metrics tracking here
        pass
