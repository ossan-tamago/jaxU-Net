import argparse
from pathlib import Path
from typing import Tuple, Any
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint
import optax
import tensorflow as tf
from tqdm import tqdm

from model import UNet


def create_output_dir(output_dir: Path) -> Tuple[Path, Path, Path]:
    ckpt_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        ckpt_dir.mkdir()
        log_dir.mkdir()

    return output_dir, ckpt_dir, log_dir


def preprocess_image(path, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image, (height - crop_size) // 2, (width - crop_size) // 2, crop_size,
                                          crop_size)
    image = tf.image.resize(image, size=(image_size, image_size), antialias=True)

    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_datasets(input_dir: Path, image_size: int = 64, batch_size: int = 64):
    preprocess_fn = partial(preprocess_image, image_size=image_size)

    train_x = tf.data.Dataset.list_files(str(input_dir / 'train' / 'images' / '*.jpg')) \
        .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(1000) \
        .batch(batch_size, drop_remainder=True)
    train_y = tf.data.Dataset.list_files(str(input_dir / 'train' / 'masks' / '*.jpg')) \
        .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(1000) \
        .batch(batch_size, drop_remainder=True)
    train_dataset = tf.data.Dataset.zip((train_x, train_y))

    val_x = tf.data.Dataset.list_files(str(input_dir / 'val' / 'images' / '*.jpg')) \
        .map(preprocess_fn) \
        .shuffle(buffer_size=10 * batch_size) \
        .batch(batch_size, drop_remainder=True)
    val_y = tf.data.Dataset.list_files(str(input_dir / 'val' / 'masks' / '*.jpg')) \
        .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(buffer_size=10 * batch_size) \
        .batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.zip((val_x, val_y))

    return train_dataset, val_dataset


class TrainState(train_state.TrainState):
    batch_stats: Any


def dice_loss(predictions, targets):
    return 1 - jnp.mean(2 * jnp.sum(predictions * targets) / (jnp.sum(predictions) + jnp.sum(targets)))


@jax.jit
def train_step(state, x, y, rng):
    def loss_fn(params):
        pred, mutated_vars = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, rng, train=True,
            mutable=['batch_stats']
        )

        loss = dice_loss(pred, y)
        return loss, mutated_vars

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutated_vars), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads,
        batch_stats=mutated_vars['batch_stats'])
    return state, loss


def train(epochs: int,
          image_size: int,
          batch_size: int,
          learning_rate: float,
          input_dir: Path,
          output_dir: Path):

    output_dir, ckpt_dir, log_dir = create_output_dir(output_dir)

    rng = jax.random.PRNGKey(0)
    rng, key_init, key_diffusion = jax.random.split(rng, 3)

    train_dataset, val_dataset = prepare_datasets(input_dir, image_size, batch_size)

    image_shape = (batch_size, image_size, image_size, 3)
    dummy = jnp.ones(image_shape, dtype=jnp.float32)

    model = UNet()
    variables = model.init(key_init, dummy, train=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    )
    rng, rng_train, rng_val = jax.random.split(rng, 3)

    for epoch in range(epochs):
        pbar = tqdm(train_dataset.as_numpy_iterator(), desc=f'Epoch {epoch}')
        for x, y in pbar:
            rng_train, key = jax.random.split(rng_train)
            state, loss = train_step(state, x, y, key)

            pbar.set_postfix({'loss': f'{loss:.5f}'})

        save_checkpoint(ckpt_dir, state, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNet training')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-i', '--input-dir', type=Path, default=Path('data'))
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('-o', '--output-dir', type=Path,
                        default=f'./outputs/{now}')

    args = parser.parse_args()

    train(**vars(args))
