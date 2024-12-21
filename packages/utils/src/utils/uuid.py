import hashlib
import uuid


def create_hash(data: str, digest_size: int = 8) -> str:
    """Creates unique hash value using BLAKE2b algorithm.

    Args:
        data (str): input string used to generate hash
        digest_size (int): The hash as a hexadecimal string of double length of `digest_size`.

    Returns:
        str: Hash value.
    """
    data = data.encode('UTF-8')
    digest = hashlib.blake2b(data, digest_size=digest_size).hexdigest()
    return digest


def create_uuid(data: str, output_format: str = 'urn', uppercase: bool = True) -> str:
    """Creates unique UUID using BLAKE2b algorithm.

    Args:
        data (str): input data used to generate UUID
        output_format (str): Output format.
            `raw` results in raw 32-char digest being returned as UUID.
            `urn` results in 36-char UUID string (32 hex values, 4 dashes)
                as defined by RFC 4122 for UUID URN Namespace
                (https://www.rfc-editor.org/rfc/rfc4122#page-4). Note that
                we only use the formatting; the UUID is not constructed from
                time seeds.
        uppercase (bool): Whether to uppercase letters in UUID.

    Returns:
        Formatted UUID.
    """
    data = data.encode('UTF-8')  # type: ignore
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()  # type: ignore
    if uppercase:
        digest = digest.upper()
    if output_format == 'raw':
        return digest
    elif output_format == 'urn':
        uuid = f'{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:]}'
        return uuid
    else:
        raise ValueError(f'Unknown argument {output_format} specified for `return_format`.')


def create_uuid_from_string(val: str) -> uuid.UUID:
    hex_string = hashlib.md5(val.encode('UTF-8')).hexdigest()
    return uuid.UUID(hex=hex_string)
