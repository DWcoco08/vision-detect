"""MQTT client for publishing vehicle damage detection results."""

import json
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttPublisher:
    """Publishes damage detection results to MQTT broker.

    Sends JSON payloads with damage type, severity score,
    and confidence for each detected damage region.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "vehicle/damage",
    ):
        """Initialize MQTT publisher.

        Args:
            broker: MQTT broker hostname or IP.
            port: MQTT broker port.
            topic: MQTT topic to publish results to.
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self._connected = False

    def connect(self) -> bool:
        """Connect to MQTT broker.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            self._connected = True
            logger.info("Connected to MQTT broker %s:%d", self.broker, self.port)
            return True
        except (ConnectionRefusedError, OSError) as e:
            logger.warning("MQTT connection failed: %s", e)
            self._connected = False
            return False

    def publish_result(
        self, damage_type: str, severity: float, confidence: float
    ) -> None:
        """Publish a single damage detection result.

        Args:
            damage_type: Detected damage class (scratch, dent, crack).
            severity: Severity score 0-100.
            confidence: Detection confidence 0-1.
        """
        if not self._connected:
            logger.warning("Not connected to MQTT broker. Skipping publish.")
            return

        payload = {
            "damage_type": damage_type,
            "severity": round(severity, 1),
            "confidence": round(confidence, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.client.publish(self.topic, json.dumps(payload))
        logger.info("Published: %s", payload)

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._connected:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            logger.info("Disconnected from MQTT broker.")
