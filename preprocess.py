import tensorflow as tf


def preprocess_image(path, image_size):
    image = tf.io.read_file(path)

    if path.endswith('.jpg'):
        image = tf.image.decode_jpeg(image, channels=3)
    elif path.endswith('.png'):
        image = tf.image.decode_png(image, channels=3)
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image, (height - crop_size) // 2, (width - crop_size) // 2, crop_size,
                                          crop_size)
    image = tf.image.resize(image, size=(image_size, image_size), antialias=True)

    return tf.clip_by_value(image / 255.0, 0.0, 1.0)