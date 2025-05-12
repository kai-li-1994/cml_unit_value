import logging
import time
from colorlog import ColoredFormatter

def logger_setup(name="trade_logger", level=logging.INFO, log_to_file=True):
    """
    Set up a color console logger with optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler with color
        stream_handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # 👈 Full date and time
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_to_file:
            file_handler = logging.FileHandler("pipeline.log")
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger

# === Logging utilities ===

# Global logger instance (imported across your modules)
logger = logger_setup()



def logger_time_info(step_name: str, start_time: float):
    """
    Log the elapsed time for a given step using the logger.
    """
    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"✅ {step_name} completed in {elapsed_time} seconds.")