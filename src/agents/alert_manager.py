import json
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from loguru import logger

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

LLM_MODEL = "z-ai/glm-5.1"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    channel: AlertChannel = AlertChannel.SLACK
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["channel"] = self.channel.value
        return d


@dataclass
class SlackConfig:
    webhook_url: str
    channel: Optional[str] = None
    username: str = "Chronos AlertBot"
    icon_emoji: str = ":chart_with_upwards_trend:"


@dataclass
class EmailConfig:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipients: list = field(default_factory=list)
    use_tls: bool = True


@dataclass
class WebhookConfig:
    url: str
    headers: dict = field(default_factory=dict)
    method: str = "POST"


class AlertManager:
    def __init__(
        self,
        slack_config: Optional[SlackConfig] = None,
        email_config: Optional[EmailConfig] = None,
        webhook_config: Optional[WebhookConfig] = None,
    ):
        self.slack_config = slack_config
        self.email_config = email_config
        self.webhook_config = webhook_config
        self._llm_client = InferenceClient(model=LLM_MODEL)
        self._alert_history: list[Alert] = []
        logger.info("AlertManager initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_natural_language_alert(
        self,
        deviation_data: dict,
        context: str = "",
    ) -> str:
        prompt = (
            "You are a senior quantitative risk analyst. Generate a clear, actionable alert "
            "message about a forecast deviation. Include: what happened, likely causes, and "
            "recommended actions. Keep it concise (2-3 sentences max)."
        )
        user_content = (
            f"Deviation data: {json.dumps(deviation_data, default=str)}\n"
            f"Market context: {context}\n"
            f"Generate alert message:"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._llm_client.chat_completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_forecast_summary(
        self,
        forecast_results: dict,
        comparison_data: Optional[dict] = None,
    ) -> str:
        prompt = (
            "You are a senior portfolio risk analyst. Summarize the forecast results in "
            "plain language. Highlight key risk factors, regime indicators, and portfolio "
            "implications. Keep it under 4 sentences."
        )
        user_content = f"Forecast results: {json.dumps(forecast_results, default=str)}\n"
        if comparison_data:
            user_content += f"Comparison vs actuals: {json.dumps(comparison_data, default=str)}\n"
        user_content += "Generate summary:"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._llm_client.chat_completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def send_alert(self, alert: Alert) -> bool:
        results = {}
        if alert.channel == AlertChannel.SLACK or alert.channel is None:
            results["slack"] = self._send_slack(alert)
        if alert.channel == AlertChannel.EMAIL or alert.channel is None:
            results["email"] = self._send_email(alert)
        if alert.channel == AlertChannel.WEBHOOK or alert.channel is None:
            results["webhook"] = self._send_webhook(alert)
        self._alert_history.append(alert)
        success = any(v for v in results.values() if v is not None)
        if success:
            logger.info(f"Alert sent via {list(results.keys())}: {alert.title}")
        else:
            logger.warning(f"Failed to send alert: {alert.title}")
        return success

    def send_deviation_alert(
        self,
        deal_name: str,
        deviation_pct: float,
        forecast_value: float,
        actual_value: float,
        context: str = "",
        severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> Alert:
        deviation_data = {
            "deal_name": deal_name,
            "deviation_pct": round(deviation_pct, 2),
            "forecast_value": round(forecast_value, 2),
            "actual_value": round(actual_value, 2),
        }
        nl_message = self.generate_natural_language_alert(deviation_data, context)
        alert = Alert(
            title=f"Forecast Deviation Alert: {deal_name}",
            message=nl_message,
            severity=severity,
            metadata=deviation_data,
        )
        self.send_alert(alert)
        return alert

    def send_regime_change_alert(
        self,
        regime_from: str,
        regime_to: str,
        confidence: float,
        indicators: dict,
    ) -> Alert:
        deviation_data = {
            "regime_transition": f"{regime_from} -> {regime_to}",
            "confidence": round(confidence, 3),
            "indicators": indicators,
        }
        nl_message = self.generate_natural_language_alert(
            deviation_data, "Regime change detected in market conditions"
        )
        alert = Alert(
            title=f"Regime Change: {regime_from} to {regime_to}",
            message=nl_message,
            severity=AlertSeverity.CRITICAL,
            metadata=deviation_data,
        )
        self.send_alert(alert)
        return alert

    def send_weekly_report(
        self,
        forecast_results: dict,
        comparison_data: Optional[dict] = None,
    ) -> Alert:
        summary = self.generate_forecast_summary(forecast_results, comparison_data)
        alert = Alert(
            title="Weekly Forecast Report",
            message=summary,
            severity=AlertSeverity.INFO,
            metadata={"forecast_results": forecast_results, "comparison": comparison_data},
        )
        self.send_alert(alert)
        return alert

    def get_alert_history(self, limit: int = 50) -> list[dict]:
        return [a.to_dict() for a in self._alert_history[-limit:]]

    def _send_slack(self, alert: Alert) -> Optional[bool]:
        if not self.slack_config:
            return None
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#f2c744",
            AlertSeverity.CRITICAL: "#ff0000",
        }
        payload = {
            "username": self.slack_config.username,
            "icon_emoji": self.slack_config.icon_emoji,
            "attachments": [
                {
                    "color": severity_colors.get(alert.severity, "#36a64f"),
                    "title": alert.title,
                    "text": alert.message,
                    "ts": int(datetime.fromisoformat(alert.timestamp).timestamp()),
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in (alert.metadata or {}).items()
                    ][:8],
                }
            ],
        }
        if self.slack_config.channel:
            payload["channel"] = self.slack_config.channel
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.slack_config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req) as resp:
                return resp.status == 200
        except (URLError, Exception) as e:
            logger.error(f"Slack send failed: {e}")
            return False

    def _send_email(self, alert: Alert) -> Optional[bool]:
        if not self.email_config or not self.email_config.recipients:
            return None
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg["From"] = self.email_config.sender_email
        msg["To"] = ", ".join(self.email_config.recipients)
        text_body = f"{alert.title}\n\n{alert.message}\n\nTimestamp: {alert.timestamp}"
        html_body = (
            f"<html><body>"
            f"<h2 style='color:{'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.WARNING else 'green'}'>{alert.title}</h2>"
            f"<p>{alert.message}</p>"
            f"<hr><p><small>Timestamp: {alert.timestamp}</small></p>"
            f"</body></html>"
        )
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        try:
            if self.email_config.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.email_config.smtp_host, self.email_config.smtp_port) as server:
                    server.starttls(context=context)
                    if self.email_config.sender_password:
                        server.login(self.email_config.sender_email, self.email_config.sender_password)
                    server.sendmail(self.email_config.sender_email, self.email_config.recipients, msg.as_string())
            else:
                with smtplib.SMTP(self.email_config.smtp_host, self.email_config.smtp_port) as server:
                    if self.email_config.sender_password:
                        server.login(self.email_config.sender_email, self.email_config.sender_password)
                    server.sendmail(self.email_config.sender_email, self.email_config.recipients, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _send_webhook(self, alert: Alert) -> Optional[bool]:
        if not self.webhook_config:
            return None
        payload = alert.to_dict()
        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            headers.update(self.webhook_config.headers)
            req = Request(self.webhook_config.url, data=data, headers=headers, method=self.webhook_config.method)
            with urlopen(req) as resp:
                return 200 <= resp.status < 300
        except (URLError, Exception) as e:
            logger.error(f"Webhook send failed: {e}")
            return False