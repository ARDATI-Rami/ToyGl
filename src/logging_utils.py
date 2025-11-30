# src/logging_utils.py

import logging

# Define ANSI escape sequences for colors
COLORS = {
    'reset': '\033[0m',      # Reset to default color
    'black': '\033[30m',     # Black text
    'red': '\033[31m',       # Red text
    'green': '\033[32m',     # Green text
    'yellow': '\033[33m',    # Yellow text
    'blue': '\033[34m',      # Blue text
    'magenta': '\033[35m',   # Magenta text
    'cyan': '\033[36m',      # Cyan text
    'white': '\033[37m',     # White text
    'bright_black': '\033[90m',  # Bright Black (also known as Grey)
    'bright_red': '\033[91m',    # Bright Red
    'bright_green': '\033[92m',  # Bright Green
    'bright_yellow': '\033[93m', # Bright Yellow
    'bright_blue': '\033[94m',   # Bright Blue
    'bright_magenta': '\033[95m',# Bright Magenta
    'bright_cyan': '\033[96m',   # Bright Cyan
    'bright_white': '\033[97m',  # Bright White
    'grey': '\033[90m',          # Grey (alias for Bright Black)
}

# Define a new log level
SUPERIOR_INFO = 25
logging.addLevelName(SUPERIOR_INFO, 'SUPERIOR_INFO')

# Clear the contents of the debug log file
log_file_path = '../debug.log'
def clear_log_file():
    # Clear the contents of the debug log file
    with open(f"{log_file_path[1:]}", 'w'):
        pass  # Opening in 'w' mode truncates the file, no need to write anything

class DebugOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG

def superior_info(self, message, *args, **kws):
    # Wrap the message with ### for SUPERIOR_INFO
    message = f"### {message} ###"
    self._log(SUPERIOR_INFO, message, args, **kws) if self.isEnabledFor(SUPERIOR_INFO) else None

# Extend the Logger class to include the superior_info method
logging.Logger.superior_info = superior_info

def dont_debug(self, message, *args, **kws):
    # This method intentionally does nothing
    pass

# Extend the Logger class to include the dont_debug method
logging.Logger.dont_debug = dont_debug

# Create a custom logger class to include the info_once method
class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._logged_once_messages = set()

    def info_once(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO', but only once.
        If the message has already been logged, it will not log it again.
        """
        if msg not in self._logged_once_messages:
            self._logged_once_messages.add(msg)
            # Add custom attribute to the record
            kwargs['extra'] = {'is_info_once': True}
            self.info(msg, *args, **kwargs)

# Register the custom logger class
logging.setLoggerClass(CustomLogger)

# Create a logger using the custom logger class
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all logs

# Create a file handler for debug logs
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(DebugOnlyFilter())  # Apply the DebugOnlyFilter

# Define a file formatter and set it to the file handler
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add file handler to the logger
logger.addHandler(file_handler)

# Create a console handler with a custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # This will handle INFO and SUPERIOR_INFO

# Custom formatter with colors, including bright green for info_once
class CustomFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelno
        if hasattr(record, 'is_info_once') and record.is_info_once:
            color = COLORS['bright_green']  # Bright green for info_once messages
        elif level == logging.INFO:
            color = COLORS['bright_white']
        elif level == SUPERIOR_INFO:
            color = COLORS['blue']
            record.msg = f"### {record.msg} ###"  # Add ### directly here for SUPERIOR_INFO messages
        else:
            color = COLORS['reset']
        message = super().format(record)
        return f"{color}{message}{COLORS['reset']}"

# Set the custom formatter to the console handler
console_formatter = CustomFormatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Add console handler to the logger
logger.addHandler(console_handler)

# Example usage
def example_usage():
    logger.info('This is a regular info message.')  # Will be displayed in bright white
    logger.superior_info('Example of Superior Info')  # Will be displayed in blue with ### prefix and suffix
    logger.debug('This debug message will go to the file.')
    logger.dont_debug('This message will not be logged anywhere.')

    # Using info_once to log only once with bright green color
    logger.info_once('This message will be logged only once in bright green.')
    logger.info_once('This message will be logged only once in bright green.')  # This will not log again
