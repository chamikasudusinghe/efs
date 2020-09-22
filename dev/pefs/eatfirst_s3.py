"""EatFirst file system for S3."""

from autorepr import autorepr
from fs.path import iteratepath
from fs.path import normpath
from fs.path import relpath
from fs_s3fs import S3FS


class EatFirstS3(S3FS):
    """Extension of S3FS class, fixing it to work with python 3k."""

    __repr__ = autorepr([], bucket=lambda self: self._bucket_name, key=lambda self: self._access_keys[0])

    def _s3path(self, path):
        """Get the absolute path to a file stored in S3."""
        path = relpath(normpath(path))
        path = self._separator.join(iteratepath(path))
        s3path = self._prefix + path
        if s3path and s3path[-1] == self._separator:
            s3path = s3path[:-1]
        return s3path
