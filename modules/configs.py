from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Callable, Optional


@dataclass
class Config:
    """
    Training Configuration Wrapper
    """
    # The directory containing the images
    images_dir: str
    #  The directory containing the labels
    labels_dir: str
    # The size to resize the images and labels to
    image_size: tuple = (1024, 2048)
    # Whether to ignore classes not in the validation set
    ignore_void_classes: bool = True
    # Default value of the index to ignore in the masks
    ignore_index: Optional[int] = None
    # Additional list of labels to ignore
    ignore_labels: Optional[list] = None
    # The transform to apply to the images and labels for training data
    train_transforms: Optional[Callable] = None
    # The transform to apply to the images and labels for validation data
    valid_transforms: Optional[Callable] = None
    # The transform to apply to the images and labels for test data
    test_transforms: Optional[Callable] = None
    # The batch size for the dataloader
    batch_size: int = 16
    # The number of workers for dataloader
    num_workers: Optional[int] = 2
    # Whether to pin memory for the dataloader
    pin_memory: Optional[bool] = False
    # Whether to shuffle the dataloader data output
    shuffle: Optional[bool] = True
    # The collate function for the dataloader
    collate_fn: Optional[Callable] = None
    # Whether to drop the last batch of the dataloader
    drop_last: Optional[bool] = False

    # Seed for reproducibility
    seed: int = 42
    # Type of device to use
    device_type: Literal["cpu", "cuda", "mps"] = "cuda"

    # Name of the model to instantiate
    model_name: str = 'unet'
    # Name of the encoder to use
    encoder_name: str = 'resnet34'
    # Number of input channels
    in_channels: int = 3
    # Depth of the encoder
    encoder_depth: int = 5
    # Weights to use for the encoder (one of imagenet or None)
    encoder_weights: str = 'imagenet'
    # Number of channels in the decoder
    decoder_channels: tuple[int] = (256, 128, 64, 32, 16)
    # Number of classification classes
    num_classes: int = 19
    # Activation function to use for the output layer
    activation: str = None
    # Path to the weights to load
    weights_path: str = None

    # Path to save outputs
    output_dir: Path = Path("outputs")
    # Relative path to save logs (tensorboard, etc.)
    log_path: str = 'logs'
    # Relative path to save model weights
    model_save_path: str = 'models'
    # Number of steps to log
    steps_per_log: int = 10
    # Number of steps to save model weights
    save_model_every: int = 10
    # Name of the experiment
    experiment_name: Optional[str] = None
    # Number of epochs to train
    num_epochs: int = 500
    # Whether to use cross entropy loss or dice loss
    use_cross_entropy_loss: bool = False
    # Learning rate
    learning_rate: float = .05
    # Weight decay
    weight_decay: float = 1e-4
    # Whether to use a learning rate scheduler
    use_scheduler: bool = True
    # Step size for the learning rate scheduler
    scheduler_step_size: int = 10
    # Gamma for the learning rate scheduler
    scheduler_gamma: float = 0.5
    # When to evaluate the model
    evaluate_model_every: int = 50
    # Whether to save the model
    eval_individual_class_metrics: bool = False


