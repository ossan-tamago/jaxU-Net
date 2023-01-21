# U-Net

U-Net written in Jax and Flax.

## Usage

### Training

```bash
python main.py
```

### Data directory structure

```bash
data
├── train
│   ├── images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   ├── masks
│   │   ├── 1.jpg
│   │   ├── 2.jpg
├── val
│   ├── images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   ├── masks
│   │   ├── 1.jpg
│   │   ├── 2.jpg
```


### Optional arguments:

    -e, --epochs - number of epochs to train (default: 1)

    --image-size - size of the image (default: 64)

    -b, --batch_size - batch size (default: 1)

    -lr, --learning_rate - learning rate (default: 0.001)
    
    -i, --input-dir - Path to the directory containing the data. (default: ./data)
    
    -o, --output-dir - Path to the directory where the model and logs should be saved. (default: ./output)

