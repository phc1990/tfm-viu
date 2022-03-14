import uuid
import os

def generate_uuid() -> str:
    return uuid.uuid1().hex

def build_file_path(dir, name: str, extension: str) -> str:
    return os.path.join(dir, name + '.' + extension)