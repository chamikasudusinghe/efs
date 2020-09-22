__version__ = "0.2.0"

from .filesystem import EFS

get_filesystem = EFS.get_filesystem

__all__ = ('EFS', 'get_filesystem', )
