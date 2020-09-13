"""The file system abstraction."""
import urllib.parse
from io import BytesIO

from flask import current_app

from .eatfirst_osfs import EatFirstOSFS
from .eatfirst_s3 import EatFirstS3


class EFS:
    """The EatFirst File system."""

    def __init__(self, *args, storage="local", **kwargs):
        """The constructor method of the filesystem abstraction."""
        self.separator = kwargs.get("separator", "/")
        self.current_file = ""
        if storage.lower() == "local":
            self.home = EatFirstOSFS(current_app.config["LOCAL_STORAGE"], create=True, *args, **kwargs)
        elif storage.lower() == "s3":
            self.home = EatFirstS3(
                current_app.config["S3_BUCKET"],
                # We always called make_public after upload, with this we do one less call to aws API
                acl="public-read",
                *args,
                **kwargs
            )
        else:
            raise RuntimeError("{} does not support {} storage".format(self.__class__.__name__, storage))

    def upload(self, path, content, content_type=None):
        """Upload a file and return its size in bytes.

        :param path: the relative path to file, including filename.
        :param content: the content to be written.
        :param content_type: Enforce content-type on destination.
        :return: size of the saved file.
        """
        path_list = path.split(self.separator)
        if len(path_list) > 1:
            self.home.makedirs(self.separator.join(path_list[:-1]), recreate=True)
        self.home.create(path, wipe=False)

        # s3fs' API sucks and we have to set the content type on construction
        # assuming this will always be called in a synchronous way, we just override the inner variable (O.o) and
        # restore it back
        content_type_key = "ContentType"
        has_upload_args = hasattr(self.home, "upload_args")
        old_content_type_exists = False
        old_content_type = None

        if has_upload_args and content_type:
            old_content_type_exists = content_type_key in self.home.upload_args
            old_content_type = self.home.upload_args.pop(content_type_key, None)
            self.home.upload_args[content_type_key] = content_type

        if isinstance(content, BytesIO):
            # TODO: The underlying library only expects bytes instead of verifying what is coming
            # maybe we should send a pr as this can result in more memory usage
            content = content.read()
        if isinstance(content, str):
            content = content.encode()
        self.home.setbytes(path, content)

        if has_upload_args:
            if old_content_type_exists:
                self.home.upload_args[content_type_key] = old_content_type
            else:
                self.home.upload_args.pop(content_type_key, None)

    def open(self, path, *args, **kwargs):
        """Open a file and return a file pointer.

        :param path: the relative path to file, including filename.
        :return: a pointer to the file.
        """
        # Maybe we should store paths as relative paths to avoid having to do this
        root_path = getattr(self.home, "_root_path", None)
        path = path.replace(root_path, "") if root_path else path
        if not self.home.exists(path):
            exp = FileNotFoundError()
            exp.filename = path
            raise exp
        return self.home.openbin(path, *args, **kwargs)

    def remove(self, path):
        """Remove a file or folder.

        :param path: the relative path to file, including filename.
        """
        if self.home.isdir(path):
            self.home.removetree(path)
        else:
            self.home.remove(path)

    def rename(self, path, new_path):
        """Rename a file.

        :param path: the relative path to file, including filename.
        :param new_path: the relative path to new file, including new filename.
        """
        self.home.move(path, new_path)

    def move(self, path, new_path):
        """Move a file.

        :param path: the relative path to file, including filename.
        :param new_path: the relative path to new file, including filename.
        """
        path_list = new_path.split(self.separator)
        if len(path_list) > 1:
            self.home.makedir(self.separator.join(path_list[:-1]), recreate=True)
        self.home.move(path, new_path, overwrite=True)

    def file_url(self, path, with_cdn=True):
        """Get a file url.

        :param path: the relative path to file, including filename.
        :param with_cdn: specify if the url should return with the cdn information, only used for images.
        """
        url = self.home.geturl(path)
        if current_app.config.get("S3_CDN_URL", None) and with_cdn:
            parsed_url = urllib.parse.urlparse(url)
            url = url.replace(parsed_url.hostname, current_app.config["S3_CDN_URL"])
        url = url.split("?")[0]  # Remove any trace of query string
        url = url.replace("http://", "https://")
        return url

    @classmethod
    def get_filesystem(cls):
        """Return an instance of the filesystem abstraction."""
        return cls(storage=current_app.config["DEFAULT_STORAGE"])
