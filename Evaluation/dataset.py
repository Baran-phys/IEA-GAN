import pathlib
import os
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


def decode_img(img):
    img = tf.image.decode_png(img, channels=1)
    return tf.cast(img, dtype=tf.float32)


def log_transform(img):
    img = tf.math.log(img + 1) / tf.math.log(256.0)
    return 2 * (img - 0.5)


def process_path(file_path, do_log_transform=True):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    if do_log_transform:
        img = log_transform(img)
    return img


def dataset_from_dir(
    data_dir,
    shuffle=True,
    seed=42,
    decode=True,
    zip_labels=False,
    return_labelnames=False,
    do_log_transform=True,
):
    """
    Assumes a directory struture like
    1.1.1/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    1.1.2/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    ...
    with the same filenames in each directory and the top-level subdirectories corresponding to the labels
    """
    data_dir = pathlib.Path(data_dir)
    dirnames = sorted(
        os.listdir(data_dir), key=lambda x: [int(i) for i in x.split(".")]
    )
    filenames = sorted(os.listdir(data_dir / "1.1.1"))

    paths = []
    for filename in filenames:
        for dirname in dirnames:
            paths.append(f"{str(data_dir)}/{dirname}/{filename}")

    ds_img = tf.data.Dataset.from_tensor_slices(paths)

    nfiles = tf.data.experimental.cardinality(ds_img).numpy()

    ds = ds_img

    if zip_labels:
        nlabels = len(dirnames)
        ds_labels = tf.data.Dataset.from_tensor_slices(
            tf.reshape(tf.cast(tf.range(40), dtype=tf.float32), (-1, 1))
        ).repeat()
        ds = tf.data.Dataset.zip((ds_img, ds_labels))

    if shuffle:
        ds = ds.shuffle(nfiles, reshuffle_each_iteration=True, seed=seed)
    if decode:
        if zip_labels:
            map_fn = lambda x, y: (process_path(x, do_log_transform=do_log_transform), y)
        else:
            map_fn = lambda x: process_path(x, do_log_transform=do_log_transform)
        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)

    ret = ds

    if return_labelnames:
        ret = (ds, dirnames)

    return ret
