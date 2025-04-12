import inspect
from typing import Any

import apprise
import numpy as np
import pandas as pd

from .time_utils import get_local_time, get_utc_time


def get_caller_filename() -> str:
    frame = inspect.stack()[1]  # [0] is this function itself, [1] is the caller
    return frame.filename  # Returns the file where the function was called


def get_current_function_arguments() -> dict[str, Any]:
    """Retrieves all arguments passed to the current function."""
    frame = inspect.currentframe().f_back
    # Get Arguments and Values
    args, _, _, localsdict = inspect.getargvalues(frame)
    # Get Dictionary of Arguments
    args_dict = {arg: localsdict[arg] for arg in args} if args else {}
    varargs_list = localsdict.get("args", None)
    if varargs_list:
        for i, arg in enumerate(varargs_list):
            args_dict[f"var_arg{i}"] = arg
    kwargs_dict = localsdict.get("kwargs", {})
    if kwargs_dict:
        args_dict |= kwargs_dict
    return args_dict


def get_parent_function_arguments() -> dict[str, Any]:
    """Retrieves all arguments passed to the parent calling function."""
    frame = inspect.currentframe().f_back.f_back  # Go up two levels in the call stack
    if frame is None:
        return {}  # Edge case: No parent function
    # Get Arguments and Values
    args, _, _, localsdict = inspect.getargvalues(frame)
    # Get Dictionary of Arguments
    args_dict = {arg: localsdict[arg] for arg in args} if args else {}
    varargs_list = localsdict.get("args", None)
    if varargs_list:
        for i, arg in enumerate(varargs_list):
            args_dict[f"var_arg{i}"] = arg
    kwargs_dict = localsdict.get("kwargs", {})
    if kwargs_dict:
        args_dict |= kwargs_dict
    return args_dict


def get_function_status_string(
    filename: str | None = None,
    start_utc_time: str | None = None,
    start_local_time: str | None = None,
    drop_value_types: list[type] = [pd.DataFrame, np.ndarray],
) -> str:
    """Creates a string of function invocation information for logging.
    By default, removes argument values that are pandas dataframes and replaces with
    placeholders to enable pretty printing.

    """
    msg = ""
    # Get Caller Filename
    if not filename:
        frame = inspect.stack()[1]  # [0] is this function itself, [1] is the caller
        filename = frame.filename
    msg += f"File: {filename}\n"

    # Get Start Timestamps (if provided)
    if start_utc_time and start_local_time:
        start_str = f"Start Time: {start_utc_time} ({start_local_time})\n"
    elif start_utc_time and not start_local_time:
        start_str = f"Start Time: {start_utc_time}\n"
    elif not start_utc_time and start_local_time:
        start_str = f"Start Time: {start_local_time}\n"
    else:
        start_str = None
    if start_str:
        msg += start_str

    # Get Current Timestamps
    utc_timestamp = get_utc_time(output_format="str")
    local_timestamp = get_local_time(output_format="str")
    current_str = f"Current Time: {utc_timestamp} ({local_timestamp})\n"
    msg += current_str

    # Get Function Arguments
    fn_args = get_parent_function_arguments()
    cleaned_fn_args: dict[str, Any] = {}
    for k, v in fn_args.items():
        for drop_type in drop_value_types:
            if isinstance(v, drop_type):
                fn_args[k] = f"{drop_type.__name__}(...)"
        cleaned_fn_args[k] = fn_args[k]
    # Format Function Arguments into String
    fn_args_str = "\n".join(f"- {k}={v}" for k, v in cleaned_fn_args.items())
    msg += f"Arguments:\n{fn_args_str}\n"
    return msg


def send_notification(
    title: str,
    message: str,
    url: str | None = None,
) -> None:
    if url:
        apobj = apprise.Apprise()
        apobj.add(url)
        apobj.notify(title=title, body=message)
