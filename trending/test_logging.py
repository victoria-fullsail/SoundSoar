import logging

# Retrieve the logger configured in your settings
logger = logging.getLogger('spotify')

# Log different levels of messages
logger.debug('This is a DEBUG message')
logger.info('This is an INFO message')
logger.warning('This is a WARNING message')
logger.error('This is an ERROR message')
logger.critical('This is a CRITICAL message')
