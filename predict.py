import argparse
from pathlib import Path
from typing import Any
from functools import partial

from flax.jax_utils import replicate, unreplicate
from flax.training import train_state
from flax.training.checkpoints import restore_checkpoint
import optax
import tensorflow as tf
from tqdm import tqdm

from model import UNet
from preprocess import preprocess_image


def save_predictions(predictions, output_dir: Path):
    for i, pred in enumerate(predictions):
        pred = (pred * 255).astype('uint8')
        img = tf.keras.preprocessing.image.array_to_img(pred)
        img.save(output_dir / f'{i}.jpg')


def prepare_predict_data(input_dir: Path, image_size: int = 64, batch_size: int = 64):
    preprocess_fn = partial(preprocess_image, image_size=image_size)

    predict_x = tf.data.Dataset.list_files(str(input_dir / 'test' / 'images' / '*.jpg')) \
        .map(preprocess_fn) \
        .shuffle(buffer_size=10 * batch_size) \
        .batch(batch_size, drop_remainder=True)

    return predict_x


def predict_step(state, batch):
    variables = unreplicate(state.params)
    pred = state.apply_fn(variables, batch, train=False)

    return pred


class TrainState(train_state.TrainState):
    batch_stats: Any


def predict(input_dir: Path, output_dir: Path, ckpt_dir: Path):
    predict_data = prepare_predict_data(input_dir)

    predict_state = TrainState.create(apply_fn=UNet().apply, params=None, tx=optax.adam(0.001))
    predict_state = restore_checkpoint(ckpt_dir, predict_state)
    predict_state = replicate(predict_state)

    predictions = []
    for batch in tqdm(predict_data.as_numpy_iterator()):
        pred = predict_step(predict_state, batch)
        predictions.append(pred)

    save_predictions(predictions, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=Path, default=Path('data'))
    parser.add_argument('-o', '--output_dir', type=Path, default=Path('output/images'))
    parser.add_argument('-c', '--ckpt_dir', type=Path, default=Path('output/models'))
    args = parser.parse_args()

    predict(args.input_dir, args.output_dir, args.ckpt_dir)