import requests
from loguru import logger


def dual_discord_sink(logs_webhook_url, alerts_webhook_url):
    def sink_function(message):
        record = message.record
        level_name = record["level"].name

        # Determine if this needs an alert
        needs_alert = level_name in ["WARNING", "ERROR", "CRITICAL"]

        # Color and emoji mapping
        color_map = {
            "TRACE": 0x808080, "DEBUG": 0x808080, "INFO": 0x0099FF,
            "SUCCESS": 0x00FF00, "WARNING": 0xFFFF00, "ERROR": 0xFF0000, "CRITICAL": 0x800080
        }

        emoji_map = {
            "TRACE": "🔍", "DEBUG": "🔍", "INFO": "ℹ️", "SUCCESS": "✅",
            "WARNING": "⚠️", "ERROR": "🚨", "CRITICAL": "💥"
        }

        # Create the embed
        embed = {
            "title": f"{emoji_map.get(level_name, '📝')} {level_name}",
            "description": f"```\n{record['message']}\n```",
            "color": color_map.get(level_name, 0x808080),
            "timestamp": record["time"].isoformat(),
            "fields": [
                {
                    "name": "Function",
                    "value": record["function"],
                    "inline": True
                },
                {
                    "name": "Location",
                    "value": f"{record['file'].name}:{record['line']}",
                    "inline": True
                }
            ]
        }

        # Add exception info if present
        if record["exception"]:
            embed["fields"].append({
                "name": "Exception",
                "value": f"```\n{record['exception']}\n```"[:1024],
                "inline": False
            })

        # ALWAYS send to logs channel (no ping)
        logs_payload = {
            "embeds": [embed],
            "username": "App Logger"
        }

        try:
            requests.post(logs_webhook_url, json=logs_payload)
        except Exception:
            logger.exception("Failed to upload to discord server")
            pass

        # Send alert to alerts channel if it's a warning/error
        if needs_alert:
            alert_embed = embed.copy()  # Same embed
            alert_embed["title"] = f"🚨 ALERT: {level_name}"

            alert_payload = {"embeds": [alert_embed], "username": "Alert Bot", "content": "@everyone"}

            try:
                requests.post(alerts_webhook_url, json=alert_payload)
            except Exception:
                logger.exception("Failed to upload to alert discord server")
                pass

    return sink_function
