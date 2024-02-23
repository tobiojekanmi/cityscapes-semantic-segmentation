
import torch
from dataclasses import dataclass
from modules.data.datamanager import CityScapesDataManager, CityScapesDataManagerConfig
import segmentation_models_pytorch as smp


@dataclass
class ModelConfig:
    """
    Model Configuration Wrapper for segmentation_models_pytorch package models.
    Supported models: Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, 
                      DeepLabV3, and DeepLabV3Plus
    """
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
    # Device to use for the model
    device: str = 'cpu'


class Model(torch.nn.Module):
    """
    Model class to load and instantiate segmentation_models_pytorch models
    """
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.config = config
        self.model = self._load_model()

    def forward(self, x):
        return self.model(x)

    def _load_model(self):
        """
        Loads new or pre-trained model
        """
        model = getattr(smp, self.config.model_name)(
            encoder_name=self.config.encoder_name,
            encoder_depth=self.config.encoder_depth,
            encoder_weights=self.config.encoder_weights,
            decoder_channels=self.config.decoder_channels,
            in_channels=self.config.in_channels,
            classes=self.config.num_classes,
            activation=self.config.activation
        )

        if self.config.weights_path is not None:
            device = torch.device(self.config.device) if isinstance(self.config.device, str) else self.config.device
            model.load_state_dict(torch.load(self.config.weights_path, map_location=device))

        return model





if __name__ == "__main__":
    config = ModelConfig(model_name='UnetPlusPlus', num_classes=19)
    model = Model(config)
    print(model)
    print('Sucessfully loaded model')

    datamanager_config = CityScapesDataManagerConfig(
        images_dir = "datasets/images/leftImg8bit/", 
        labels_dir = "datasets/labels/gtFine", 
        image_size = (128, 128),
        batch_size = 2,
        shuffle = True
    )
    datamanager = CityScapesDataManager(datamanager_config)
    train_dataloader = datamanager.get_train_dataloader()
    train_batch = next(iter(train_dataloader))
    predictions = model(train_batch[0])

    print(f"Image: {train_batch[0].shape}\n", 
          f"Label: {train_batch[1].shape}\n", 
          f"Predictions: {predictions.shape}")
    print('Sucessfully predicted on batch')