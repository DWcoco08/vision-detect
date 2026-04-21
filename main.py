"""Vehicle Damage Detection and Severity Estimation — Main Pipeline.

End-to-end inference: load image → detect damage → estimate severity
→ visualize results → optionally publish via MQTT.

Usage:
    python main.py --image test_car.jpg
    python main.py --image test_car.jpg --output result.jpg
    python main.py --image test_car.jpg --mqtt --mqtt-broker 192.168.1.100
"""

import argparse
import logging
import sys

import cv2

from models.yolo_model import DamageDetector
from models.severity_model import SeverityPredictor
from mqtt.mqtt_client import MqttPublisher
from utils.preprocessing import crop_detection, load_image
from utils.visualization import draw_detections

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run full damage detection and severity estimation pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    # 1. Load models
    logger.info("Loading YOLO model: %s", args.yolo_weights)
    detector = DamageDetector(args.yolo_weights, confidence=args.confidence)

    logger.info("Loading severity model: %s", args.severity_weights)
    predictor = SeverityPredictor(args.severity_weights, device=args.device)

    # 2. Load image
    logger.info("Reading image: %s", args.image)
    image = load_image(args.image)

    # 3. Detect damage regions
    detections = detector.detect(image)
    logger.info("Found %d damage region(s)", len(detections))

    if not detections:
        print("\n=== Vehicle Damage Detection Results ===")
        print(f"Image: {args.image}")
        print("Detections: 0 — No damage detected.")
        print("=" * 40)
        return

    # 4. Predict severity for each detection
    severities = []
    for det in detections:
        crop = crop_detection(image, det.bbox)
        severity = predictor.predict(crop)
        severities.append(severity)

    # 5. Print results to console
    _print_results(args.image, detections, severities)

    # 6. Visualize — draw bboxes + labels on image
    annotated = draw_detections(image, detections, severities)

    # 7. Save or display
    if args.output:
        cv2.imwrite(args.output, annotated)
        logger.info("Saved annotated image to %s", args.output)
    else:
        cv2.imshow("Vehicle Damage Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 8. MQTT publish (optional)
    if args.mqtt:
        _publish_mqtt(args, detections, severities)


def _print_results(image_path: str, detections: list, severities: list[float]) -> None:
    """Print formatted detection results to console."""
    print("\n=== Vehicle Damage Detection Results ===")
    print(f"Image: {image_path}")
    print(f"Detections: {len(detections)}\n")

    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
        print(
            f"[{i}] {det.class_name:<8} "
            f"(conf: {det.confidence:.1%})  "
            f"Severity: {sev_text}  "
            f"bbox: [{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]"
        )

    print("=" * 40)


def _publish_mqtt(
    args: argparse.Namespace, detections: list, severities: list[float]
) -> None:
    """Publish all detection results via MQTT."""
    publisher = MqttPublisher(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
    )

    if not publisher.connect():
        logger.warning("MQTT publish skipped — broker unavailable.")
        return

    for det, sev in zip(detections, severities):
        publisher.publish_result(det.class_name, sev, det.confidence)

    publisher.disconnect()
    logger.info("Published %d result(s) via MQTT.", len(detections))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vehicle Damage Detection and Severity Estimation"
    )

    # Required
    parser.add_argument("--image", required=True, help="Path to input image")

    # Model paths
    parser.add_argument(
        "--yolo-weights",
        default="weights/best.pt",
        help="Path to YOLO model weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--severity-weights",
        default="weights/severity.pth",
        help="Path to severity model weights (default: weights/severity.pth)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for severity model (default: cpu)",
    )

    # Output
    parser.add_argument(
        "--output",
        help="Save annotated image to path (default: display window)",
    )

    # MQTT options
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Enable MQTT publishing",
    )
    parser.add_argument(
        "--mqtt-broker",
        default="localhost",
        help="MQTT broker address (default: localhost)",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--mqtt-topic",
        default="vehicle/damage",
        help="MQTT topic (default: vehicle/damage)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
