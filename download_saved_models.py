"""Download model weights."""
import zipfile

try:
    from torch.utils.model_zoo import _download_url_to_file
except ImportError:
    try:
        from torch.hub import download_url_to_file as _download_url_to_file
    except ImportError:
        from torch.hub import _download_url_to_file


def unzip(source_filename, dest_dir):
    """Unzips an archive."""
    with zipfile.ZipFile(source_filename) as zfile:
        zfile.extractall(path=dest_dir)


if __name__ == '__main__':
    URL_LINK = 'https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1'
    _download_url_to_file(URL_LINK, 'saved_models.zip', None, True)
    unzip('saved_models.zip', '.')
