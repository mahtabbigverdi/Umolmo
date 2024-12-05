import os
from multiprocessing import Pool


def add_images_par(images, group, n_procs):
    with Pool(n_procs) as pool:
        for ex in pool.imap(_add_image, [(img, group) for img in images]):
            yield ex


def _add_image(args):
    return add_image(*args)


def add_image(image_file_or_bytes, group):
    image_dir = os.environ["image_dir"]
    raise NotImplementedError()