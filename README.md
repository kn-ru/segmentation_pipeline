## Training Pipeline for Segmentation Models using segmentation_models_pytorch

This project includes a script for training a segmentation model using the PyTorch library and segmentation_models_pytorch ([smp](https://github.com/qubvel/segmentation_models.pytorch)).

## Features
 - Utilizes pre-trained encoders from the smp library.
 - Supports various segmentation model architectures.
 - Easy customization of training parameters via command-line arguments.
 - Automatic splitting of data into training and validation sets.
 - Integration with TensorBoard for logging and visualizing training metrics.

## Requirements

To run the script, the following libraries need to be installed:
    PyTorch
    torchvision
    segmentation_models_pytorch
    split-folders
    tensorboard (for TensorBoard logging)

## Usage

The script supports the following command-line arguments:

    --data: Path to the data for processing (default: empty).
    --output_folder: Path to the output data (default: './outputs').
    --epochs: Number of training epochs (default: 10).
    --batch_size: Batch size (default: 4).
    --learning_rate: Learning rate (default: 0.0003).
    --model_name: Name of the model from smp (default: 'Unet').
    --encoder_name: Name of the encoder (default: 'efficientnet-b0').
    --activation: Activation function (default: 'softmax2d').
    --classes: List of classes, separated by commas (default: 'tower').
    --device: GPU number to use (default: 0).

## Example Run

```
python3 train.py --data /path/to/data --epochs 20 --batch_size 8 --learning_rate 0.001 --model_name Unet --encoder_name resnet34 --activation softmax2d --classes class1,class2 --device 0
```

## Data Structure

Data should be pre-processed and have the following structure:
```
/path/to/data/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── masks/
        ├── img1.png
        ├── img2.png
        └── ...
```
After running the script, a directory train_val_test will be created at the same level as the specified data directory, with divisions into training and validation sets.

## Monitoring and Background Execution

**TensorBoard**: To visualize training logs, use TensorBoard by navigating to the output folder and running tensorboard --logdir=runs. You can also upload these logs to tensorboard.dev for sharing and remote access.

**Running in Background:** For continuous training, you can use pm2 to run the script in the background. Create a main.sh bash script with the command python3 train.py and your desired arguments. Then, use pm2 start main.sh to initiate the training process in the background.
```
pm2 start main.sh --name "train segmentation model" 
```