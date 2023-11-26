# -*- coding: utf-8 -*-

import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from dataset import *
import splitfolders
import argparse
import os

def main():
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description='Обучение сегментационной модели с использованием smp')
    parser.add_argument('--data', type=str, default='', help='путь к данным для обработки')
    parser.add_argument('--output_folder', type=str, default='./outputs', help='путь к выходным данным')
    parser.add_argument('--epochs', type=int, default=10, help='количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=4, help='размер батча')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='скорость обучения')
    parser.add_argument('--model_name', type=str, default='Unet', help='название модели из smp (например, Unet, FPN и т.д.)')
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0', help='название энкодера')
    parser.add_argument('--activation', type=str, default='softmax2d', help='функция активации')
    parser.add_argument('--classes', type=str, default='tower', help='список классов, разделенных запятыми')


    args = parser.parse_args()

    # Печать параметров
    print('\n------------------------------------------------------------------')
    print("\nКонфигурация параметров:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('------------------------------------------------------------------\n')

    classes = ['background'] + args.classes.split(',') if args.classes else ['background', 'object']

    # Создание модели
    model = getattr(smp, args.model_name)(
        encoder_name=args.encoder_name, 
        encoder_weights='imagenet', 
        in_channels=3, 
        classes=len(classes),
        activation=args.activation
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Параметры обучения
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Подготовка данных
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, 'imagenet')

    # Создание папки с результатами
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)


    # Обработка данных
    parent_folder = os.path.dirname(args.data)
    data_folder = os.path.join(parent_folder, "train_val_test")
    if not os.path.exists(data_folder):
        print("Разделение данных...")
        os.makedirs(data_folder, exist_ok=True)
        splitfolders.ratio(args.data, output=data_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False)
        print(f"Данные разделены и сохранены в {data_folder}")
    else:
        print(f"Папка {data_folder} уже существует. Пропускаем разделение данных.")
    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")
    x_train_dir = os.path.join(train_dir, "images")
    y_train_dir = os.path.join(train_dir, "masks")
    x_val_dir = os.path.join(val_dir, "images")
    y_val_dir = os.path.join(val_dir, "masks")

    # Создание набора данных
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

    # Создание загрузчика
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Эпохи обучения и валидации
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
    # Цикл обучения
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(output_folder, f'model_{args.encoder_name}_epochs{epoch}_batch{args.batch_size}.pth'))
            print('save best model')
            

if __name__ == '__main__':
    main()