import logging

# Configure the logger
logger = logging.getLogger("geoPipeline")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("geoPipeline.log")
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Attach the handler to the logger
logger.addHandler(file_handler)

# # Console output
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# # Example usage
# if __name__ == "__main__":
#     logger.info("Logger is configured and ready to use.")