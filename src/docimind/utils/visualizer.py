import numpy as np
from typing import List, Dict, Any

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class Visualizer:
    """
    OpenCV Bounding Box Visualizer to overlay OCR text detections on image.
    """

    @staticmethod
    def draw_bounding_boxes(
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Draws bounding box rectangles and optional text labels on the image.
        """
        annotated = image_bgr.copy()
        if not HAS_CV2:
            return annotated

        for det in detections:
            bbox = det.get("bbox", [])
            conf = det.get("confidence", 0.0)
            text = det.get("text", "")

            if len(bbox) < 4:
                continue

            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))

            if conf >= 0.70:
                color = (0, 255, 0)
            elif conf >= 0.40:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            if draw_labels and text:
                x_min = min([pt[0] for pt in bbox])
                y_min = min([pt[1] for pt in bbox])
                label = f"{text[:15]} ({int(conf * 100)}%)"

                cv2.putText(
                    annotated,
                    label,
                    (max(0, x_min), max(15, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated,
                    label,
                    (max(0, x_min), max(15, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

        return annotated
