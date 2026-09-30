import contextlib
import io
import unittest
from pathlib import Path

from app.services.logging.gateway import GatewayLog, GatewayLogPolicy
from tests.gateway.support import temporary_directory


class GatewayLoggingTests(unittest.TestCase):
    def test_global_gateway_log_redacts_secrets(self):
        with temporary_directory() as directory:
            path = Path(directory) / "logs" / "gateway.log"
            logger = GatewayLog(path)
            with contextlib.redirect_stdout(io.StringIO()):
                logger.write("connection failed token=very-secret-value")
            content = path.read_text(encoding="utf-8")
            self.assertIn("[REDACTED]", content)
            self.assertNotIn("very-secret-value", content)

    def test_global_gateway_log_rotates_and_keeps_numbered_backups(self):
        with temporary_directory() as directory:
            path = Path(directory) / "logs" / "gateway.log"
            logger = GatewayLog(
                path, policy=GatewayLogPolicy(max_bytes=1024, backup_count=2)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                logger.write("x" * 800)
                logger.write("y" * 800)
                logger.write("z" * 800)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_name("gateway.log.1").exists())
            self.assertTrue(path.with_name("gateway.log.2").exists())


if __name__ == "__main__":
    unittest.main()
