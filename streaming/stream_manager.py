from typing import Dict, Any

class StreamManager:
    """Manages optional IP video streaming configuration (IP, Port, Status)."""
    def __init__(self, ip: str = "192.168.1.100", port: int = 5000):
        self.ip = ip
        self.port = port
        self.is_streaming = False

    def start_stream(self, target_ip: str = None, target_port: int = None) -> bool:
        if target_ip:
            self.ip = target_ip
        if target_port:
            self.port = target_port
        self.is_streaming = True
        return True

    def stop_stream(self) -> None:
        self.is_streaming = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "streaming": self.is_streaming,
            "ip": self.ip,
            "port": self.port,
            "status_text": "CONNECTED" if self.is_streaming else "DISCONNECTED",
        }
