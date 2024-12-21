import time


def get_utc_time(output_format: str = "iso") -> str:
    """Get current UTC time."""
    # Time in seconds since epoch as float
    t = time.time()
    # Convert epoch time to UTC time tuple (time.struct_time object)
    utc_time = time.gmtime(t)
    # Format UTC time for output
    if output_format == "str":
        return time.strftime("%Y-%m-%d %H:%M %Z", utc_time)
    if output_format == "iso":
        return time.strftime("%Y-%m-%dT%H:%M:%S", utc_time)
    else:
        # Returns a time.struct_time object
        return utc_time


def get_local_time(output_format: str = "iso") -> str:
    """Get current UTC time."""
    # Time in seconds since epoch as float
    t = time.time()
    # Convert epoch time to UTC time tuple (time.struct_time object)
    local_time = time.localtime(t)
    # Format UTC time for output
    if output_format == "str":
        return time.strftime("%Y-%m-%d %H:%M %Z", local_time)
    if output_format == "iso":
        return time.strftime("%Y-%m-%dT%H:%M:%S", local_time)
    else:
        # Returns a time.struct_time object
        return local_time
