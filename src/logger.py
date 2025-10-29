import logging
import os
from datetime import datetime

#==========================#
#----Config for Logging----#
#==========================#

TIMESTAMP = datetime.now().strftime('%d_%b_%Y_%H_%M')
LOG_FILE = f'{TIMESTAMP}.log'
# log_file_dir_path = "./src/.logs"

# LOG_FILE_PATH = os.path.join(os.getcwd(),log_file_dir_path, LOG_FILE)

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Started logging")