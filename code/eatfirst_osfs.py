from autorepr import autorepr
from fs.osfs import OSFS


class EatFirstOSFS(OSFS):
    """Simple wrapper to have a better repr."""

    __repr__ = autorepr(['root_path', 'dir_mode:o'])
