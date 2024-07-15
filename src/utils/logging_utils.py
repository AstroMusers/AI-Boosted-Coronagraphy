import logging

def setup_logging(logfile='project.log'):
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def log_message(message):
    logging.info(message)
