import logging
import os
from datetime import datetime

LOG_FILE = F'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
LOG_PATH = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.mkdir(LOG_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)