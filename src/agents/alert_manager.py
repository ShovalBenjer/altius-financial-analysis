"""Alert manager for sending notifications via Slack, Email, and Webhook."""
import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path


class AlertManager:
    """Manages alert delivery across multiple channels."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize AlertManager with configuration."""
        self.config = self._load_config(config_path)
        self.alerts_config = self.config.get("alerts", {})
        self.alert_history_path = Path(self.config.get("data", {}).get(
            "history_path", "results/history")) / "alerts.csv"
        self.alert_history_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Send alert through specified channels.

        Args:
            title: Alert title
            message: Alert message body
            severity: One of 'info', 'warning', 'critical'
            channels: List of channels to use ('email', 'slack', 'webhook')
                     If None, uses all enabled channels
            metadata: Additional context data

        Returns:
            Dict mapping channel name to success status
        """
        results = {}

        if channels is None:
            channels = []
            if self.alerts_config.get("email", {}).get("enabled", False):
                channels.append("email")
            if self.alerts_config.get("slack", {}).get("enabled", False):
                channels.append("slack")
            if self.alerts_config.get("webhook", {}).get("enabled", False):
                channels.append("webhook")

        for channel in channels:
            try:
                if channel == "email":
                    results[channel] = self._send_email(title, message, severity, metadata)
                elif channel == "slack":
                    results[channel] = self._send_slack(title, message, severity, metadata)
                elif channel == "webhook":
                    results[channel] = self._send_webhook(title, message, severity, metadata)
                else:
                    logger.warning(f"Unknown alert channel: {channel}")
                    results[channel] = False
            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")
                results[channel] = False

        self._log_alert(title, message, severity, channels, results, metadata)
        return results

    def _send_email(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send email alert via SMTP."""
        email_config = self.alerts_config.get("email", {})
        if not email_config.get("enabled"):
            return False

        smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
        smtp_port = email_config.get("smtp_port", 587)
        sender = email_config.get("sender_email")
        recipients = email_config.get("recipient_emails", [])
        use_tls = email_config.get("use_tls", True)

        if not sender or not recipients:
            logger.warning("Email not configured: missing sender or recipients")
            return False

        msg = MIMEMultipart()
        msg["Subject"] = f"[{severity.upper()}] {title}"
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)

        # Build HTML email body
        html = f"""
        <html>
        <body>
            <h2>{title}</h2>
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Time:</strong> {datetime.utcnow().isoformat()}Z</p>
            <hr>
            <pre>{message}</pre>
            <hr>
        """
        if metadata:
            html += "<h3>Details</h3><pre>" + json.dumps(metadata, indent=2) + "</pre>"
        html += "</body></html>"

        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.starttls()
                # Use env vars for credentials if not in config
                password = os.getenv("SMTP_PASSWORD") or email_config.get("password")
                username = os.getenv("SMTP_USERNAME") or email_config.get("username")
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
            logger.success(f"Email sent to {recipients}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _send_slack(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send Slack alert via webhook."""
        slack_config = self.alerts_config.get("slack", {})
        if not slack_config.get("enabled"):
            return False

        webhook_url = slack_config.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        # Slack color based on severity
        colors = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
        color = colors.get(severity, "#cccccc")

        payload = {
            "attachments": [{
                "color": color,
                "title": title,
                "text": message,
                "fields": [
                    {"title": "Severity", "value": severity, "short": True},
                    {"title": "Timestamp", "value": datetime.utcnow().isoformat() + "Z", "short": True}
                ]
            }]
        }

        if metadata:
            payload["attachments"][0]["fields"].append({
                "title": "Details",
                "value": f"```{json.dumps(metadata, indent=2)}```",
                "short": False
            })

        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.success("Slack alert sent")
            return True
        else:
            logger.error(f"Slack webhook failed: {response.status_code} {response.text}")
            return False

    def _send_webhook(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send generic webhook alert."""
        webhook_config = self.alerts_config.get("webhook", {})
        if not webhook_config.get("enabled"):
            return False

        url = webhook_config.get("url") or os.getenv("WEBHOOK_URL")
        if not url:
            logger.warning("Webhook URL not configured")
            return False

        headers = webhook_config.get("headers", {})
        payload = {
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code in (200, 201, 202):
            logger.success("Webhook alert sent")
            return True
        else:
            logger.error(f"Webhook failed: {response.status_code} {response.text}")
            return False

    def _log_alert(
        self,
        title: str,
        message: str,
        severity: str,
        channels: List[str],
        results: Dict[str, bool],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Log alert to history file."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "title": title,
            "message": message[:500],
            "severity": severity,
            "channels": ",".join(channels),
            "results": json.dumps(results),
            "metadata": json.dumps(metadata or {})
        }

        file_exists = self.alert_history_path.exists()
        with open(self.alert_history_path, 'a') as f:
            if not file_exists:
                f.write(",".join(record.keys()) + "\n")
            f.write(",".join(str(v) for v in record.values()) + "\n")

    def send_forecast_deviation_alert(
        self,
        deal_name: str,
        forecast_value: float,
        actual_value: float,
        deviation_pct: float,
        threshold_pct: float
    ) -> Dict[str, bool]:
        """Specialized alert for forecast deviations."""
        severity = "critical" if abs(deviation_pct) > threshold_pct * 2 else "warning"
        title = f"Forecast Deviation Alert: {deal_name}"
        message = (
            f"Actual value deviates significantly from forecast.\n\n"
            f"Deal: {deal_name}\n"
            f"Forecast: ${forecast_value:,.2f}\n"
            f"Actual: ${actual_value:,.2f}\n"
            f"Deviation: {deviation_pct:+.1f}%\n"
            f"Threshold: {threshold_pct:.1f}%"
        )
        return self.send_alert(title, message, severity, metadata={
            "deal_name": deal_name,
            "forecast": forecast_value,
            "actual": actual_value,
            "deviation_pct": deviation_pct,
            "threshold_pct": threshold_pct,
            "alert_type": "forecast_deviation"
        })

    def send_regime_change_alert(
        self,
        deal_name: str,
        old_regime: str,
        new_regime: str,
        metrics: Dict[str, float]
    ) -> Dict[str, bool]:
        """Alert for detected regime changes."""
        title = f"Regime Change Detected: {deal_name}"
        message = (
            f"The forecasting regime for this deal has changed.\n\n"
            f"Deal: {deal_name}\n"
            f"Previous Regime: {old_regime}\n"
            f"New Regime: {new_regime}\n"
        )
        return self.send_alert(title, message, "warning", metadata={
            "deal_name": deal_name,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "metrics": metrics,
            "alert_type": "regime_change"
        })

    def send_forecast_summary_report(
        self,
        summary_stats: Dict[str, Any],
        alerts_triggered: int
    ) -> Dict[str, bool]:
        """Send periodic forecast summary report."""
        title = "Auto-Forecast Agent: Periodic Summary"
        message = (
            f"Forecast run completed.\n\n"
            f"Deals forecasted: {summary_stats.get('deals_count', 0)}\n"
            f"Total AUM: ${summary_stats.get('total_aum', 0):,.2f}\n"
            f"Mean forecast return: {summary_stats.get('mean_return', 0):.1f}%\n"
            f"Alerts triggered: {alerts_triggered}\n"
            f"Run timestamp: {datetime.utcnow().isoformat()}Z"
        )
        return self.send_alert(title, message, "info", metadata={
            "summary": summary_stats,
            "alerts_triggered": alerts_triggered,
            "alert_type": "forecast_summary"
        })


def get_alert_manager(config_path: str = "config.json") -> AlertManager:
    """Factory function to create AlertManager instance."""
    return AlertManager(config_path)