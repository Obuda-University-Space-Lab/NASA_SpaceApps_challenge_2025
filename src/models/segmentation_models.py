"""
AI-based segmentation models for satellite imagery
Implements various deep learning architectures for image segmentation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    PyTorch implementation
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class DoubleConv(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DeepLabV3Plus:
    """
    DeepLabV3+ implementation using TensorFlow/Keras
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def atrous_spatial_pyramid_pooling(self, inputs):
        """ASPP module"""
        dims = tf.keras.backend.int_shape(inputs)
        
        # Image pooling
        image_pooling = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        image_pooling = tf.keras.layers.Reshape((1, 1, dims[-1]))(image_pooling)
        image_pooling = tf.keras.layers.Conv2D(256, (1, 1), padding='same', 
                                              use_bias=False)(image_pooling)
        image_pooling = tf.keras.layers.BatchNormalization()(image_pooling)
        image_pooling = tf.keras.layers.Activation('relu')(image_pooling)
        image_pooling = tf.keras.layers.UpSampling2D((dims[1], dims[2]), 
                                                    interpolation='bilinear')(image_pooling)
        
        # 1x1 convolution
        conv_1x1 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', 
                                         use_bias=False)(inputs)
        conv_1x1 = tf.keras.layers.BatchNormalization()(conv_1x1)
        conv_1x1 = tf.keras.layers.Activation('relu')(conv_1x1)
        
        # Atrous convolutions
        conv_3x3_1 = tf.keras.layers.Conv2D(256, (3, 3), dilation_rate=6,
                                           padding='same', use_bias=False)(inputs)
        conv_3x3_1 = tf.keras.layers.BatchNormalization()(conv_3x3_1)
        conv_3x3_1 = tf.keras.layers.Activation('relu')(conv_3x3_1)
        
        conv_3x3_2 = tf.keras.layers.Conv2D(256, (3, 3), dilation_rate=12,
                                           padding='same', use_bias=False)(inputs)
        conv_3x3_2 = tf.keras.layers.BatchNormalization()(conv_3x3_2)
        conv_3x3_2 = tf.keras.layers.Activation('relu')(conv_3x3_2)
        
        conv_3x3_3 = tf.keras.layers.Conv2D(256, (3, 3), dilation_rate=18,
                                           padding='same', use_bias=False)(inputs)
        conv_3x3_3 = tf.keras.layers.BatchNormalization()(conv_3x3_3)
        conv_3x3_3 = tf.keras.layers.Activation('relu')(conv_3x3_3)
        
        # Concatenate all branches
        concat = tf.keras.layers.Concatenate()([
            image_pooling, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3
        ])
        
        # Final convolution
        concat = tf.keras.layers.Conv2D(256, (1, 1), padding='same', 
                                       use_bias=False)(concat)
        concat = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Activation('relu')(concat)
        concat = tf.keras.layers.Dropout(0.1)(concat)
        
        return concat
    
    def build_model(self):
        """Build DeepLabV3+ model"""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Encoder - using MobileNetV2 as backbone
        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=inputs, weights='imagenet', include_top=False
        )
        
        # Get intermediate features
        low_level_features = backbone.get_layer('block_3_expand_relu').output
        
        # ASPP
        x = self.atrous_spatial_pyramid_pooling(backbone.output)
        
        # Decoder
        x = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
        
        # Process low-level features
        low_level_features = tf.keras.layers.Conv2D(48, (1, 1), padding='same',
                                                   use_bias=False)(low_level_features)
        low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
        low_level_features = tf.keras.layers.Activation('relu')(low_level_features)
        
        # Concatenate
        x = tf.keras.layers.Concatenate()([x, low_level_features])
        
        # Final decoder layers
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', 
                                  use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', 
                                  use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Final upsampling and classification
        x = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
        
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), 
                                        activation='sigmoid' if self.num_classes == 1 else 'softmax')(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model


class SegmentationTrainer:
    """
    Training utilities for segmentation models
    """
    
    def __init__(self, model_type: str = 'unet'):
        self.model_type = model_type
        self.model = None
        
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                              tf.keras.backend.sum(y_pred_f) + smooth)
    
    def dice_loss(self, y_true, y_pred):
        """Dice loss function"""
        return 1 - self.dice_coefficient(y_true, y_pred)
    
    def iou_metric(self, y_true, y_pred, smooth=1e-6):
        """Intersection over Union metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def build_keras_unet(self, input_shape: Tuple[int, int, int], 
                        num_classes: int = 1):
        """Build U-Net model using Keras"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Encoder
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bridge
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)
        
        # Decoder
        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output
        outputs = tf.keras.layers.Conv2D(num_classes, 1, 
                                        activation='sigmoid' if num_classes == 1 else 'softmax')(conv9)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, model, learning_rate: float = 0.001):
        """Compile model with appropriate loss and metrics"""
        if self.model_type == 'deeplabv3plus':
            loss = 'binary_crossentropy' if model.output_shape[-1] == 1 else 'categorical_crossentropy'
        else:
            loss = self.dice_loss
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[self.dice_coefficient, self.iou_metric, 'accuracy']
        )
        return model
    
    def train_model(self, model, train_generator, val_generator, 
                   epochs: int = 50, patience: int = 10):
        """Train segmentation model"""
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7)
        ]
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history