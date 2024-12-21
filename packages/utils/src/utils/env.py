def load_environment() -> None:
    """Finds .env files in project and loads them to os.environ."""
    import warnings

    from dotenv import find_dotenv, load_dotenv

    try:
        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        warnings.warn("Could not find `.env` file.")
