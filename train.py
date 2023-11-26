import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from dataset import *
import splitfolders
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Train a segmentation model using smp')
    parser.add_argument('--data', type=str, default='', help='path to the data for processing')
    parser.add_argument('--output_folder', type=str, default='./outputs', help='path to the output data')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--model_name', type=str, default='Unet', help='model name from smp (e.g., Unet, FPN, etc.)')
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0', help='name of the encoder')
    parser.add_argument('--activation', type=str, default='softmax2d', help='activation function')
    parser.add_argument('--classes', type=str, default='tower', help='list of classes, separated by commas')
    parser.add_argument('--device', type=int, default=0, help='GPU number to use (default 0)')

    args = parser.parse_args()

    # Print parameters
    print('\n------------------------------------------------------------------')
    print("\nConfiguration parameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('------------------------------------------------------------------\n')

    classes = ['background'] + args.classes.split(',') if args.classes else ['background', 'object']

    # Model creation
    model = getattr(smp, args.model_name)(
        encoder_name=args.encoder_name, 
        encoder_weights='imagenet', 
        in_channels=3, 
        classes=len(classes),
        activation=args.activation
    )

    # Setting the device
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')
    model.to(device)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Data preparation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, 'imagenet')

    # Create output folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Data processing
    parent_folder = os.path.dirname(args.data)
    data_folder = os.path.join(parent_folder, "train_val_test")
    if not os.path.exists(data_folder):
        print("Splitting data...")
        os.makedirs(data_folder, exist_ok=True)
        splitfolders.ratio(args.data, output=data_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False)
        print(f"Data split and saved in {data_folder}")
    else:
        print(f"Folder {data_folder} already exists. Skipping data splitting.")
    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")
    x_train_dir = os.path.join(train_dir, "images")
    y_train_dir = os.path.join(train_dir, "masks")
    x_val_dir = os.path.join(val_dir, "images")
    y_val_dir = os.path.join(val_dir, "masks")

    # Dataset creation
    train_dataset = SegmentationDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )
    valid_dataset = SegmentationDataset(
        x_val_dir,
        y_val_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    # DataLoader creation
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    # TensorBoard setup
    writer = SummaryWriter()

    # Training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0
    # Training loop
    print('\n')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_logs['dice_loss'], epoch)
        writer.add_scalar('Train/IoU', train_logs['iou_score'], epoch)
        writer.add_scalar('Valid/Loss', valid_logs['dice_loss'], epoch)
        writer.add_scalar('Valid/IoU', valid_logs['iou_score'], epoch)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(output_folder, f'model_{args.encoder_name}_epochs{epoch}_batch{args.batch_size}.pth'))
            print('Saving best model')

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()