import logging
from pathlib import Path
from typing import Any


class LazyFileLogger(logging.Logger):
    """Wrapper around logger that lazily initializes file handler on first log message.
    This prevents log files from being created when the logger is created but never used.
    """

    def __init__(self, name: str, log_file: str | Path | None, level=logging.DEBUG) -> None:
        super().__init__(name, level)
        self.log_file = Path(log_file).as_posix() if log_file else None
        self.handler_initialized = False

    def _initialize_file_handler(self) -> None:
        if not self.handler_initialized and self.log_file:
            # Create a file handler and set the formatter
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(funcName)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.addHandler(file_handler)
            self.handler_initialized = True

    def _log(self, level, msg, args, **kwargs):
        "Override the default low-level log method to initialize the file handler lazily."
        self._initialize_file_handler()
        super()._log(level, msg, args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        # Return a dictionary of the state that should be pickled
        state = self.__dict__.copy()  # Copy current state
        # Remove handlers to avoid pickling them (they cannot be pickled)
        state["handlers"] = []
        state["handler_initialized"] = False
        return state

    def __setstate__(self, state) -> None:
        # Restore the state from the pickled data
        self.__dict__.update(state)  # Update state
        # Reinitialize the handler if needed
        if state["handler_initialized"]:
            self._initialize_file_handler()


def setup_logger(
    name: str | None = None, level: str = logging.DEBUG, log_file: str | Path | None = None
) -> LazyFileLogger:
    return LazyFileLogger(name=name, level=level, log_file=log_file)
