import argparse
from modules.configs import Config
from modules.engine import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the CityScapes dataset.")
    parser.add_argument("--images_dir", type=str, required=True, help="SeThe directory containing the images.")
    parser.add_argument("--labels_dir", type=str, required=True, help="The directory containing the labels.")
    parser.add_argument("--image_height", type=int, default=256, help="The height to resize the images.")
    parser.add_argument("--image_width", type=int, default=256, help="The width to resize the images.")
    parser.add_argument("--in_channels", type=int, default=3, help="The number of input channels.")
    parser.add_argument("--ignore_void_classes", type=bool, default=True, help="Whether to ignore classes not in the validation set.")
    parser.add_argument("--ignore_index", type=int, default=None, help="Default value of the index to ignore in the masks.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for the dataloader.")
    parser.add_argument("--num_workers", type=int, default=2, help="The number of workers for the dataloader.")
    parser.add_argument("--pin_memory", type=bool, default=False, help="Whether to pin memory for the dataloader.")
    parser.add_argument("--drop_last", type=bool, default=False, help="Whether to drop the last batch of the dataloader.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the dataloader data output.")
    parser.add_argument("--model_name", type=str, default="Unet", help="The name of the model to use.")
    parser.add_argument("--encoder_name", type=str, default="resnet34", help="The name of the encoder to use.")
    parser.add_argument("--encoder_depth", type=int, default=5, help="The depth of the encoder.")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="The weights to use for the encoder.")
    parser.add_argument("--num_classes", type=int, default=19, help="The number of classes in the dataset.")
    parser.add_argument("--weights_path", type=str, default=None, help="The path to the model weights to load.")
    parser.add_argument("--num_epochs", type=int, default=50, help="The number of epochs to train.")
    parser.add_argument("--use_cross_entropy_loss", type=bool, default=False, help="Whether to use cross entropy loss or dice loss.")
    parser.add_argument("--device_type", type=str, default="cuda", help="The type of device to use.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="The learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay for the optimizer.")
    parser.add_argument("--use_scheduler", type=bool, default=True, help="Whether to use a learning rate scheduler.")
    parser.add_argument("--scheduler_step_size", type=int, default=10, help="Step size for the learning rate scheduler.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--evaluate_model_every", type=int, default=50, help="When to evaluate the model.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The relative path to save outputs.")
    parser.add_argument("--log_path", type=str, default="logs", help="The relative path to save logs.")
    parser.add_argument("--experiment_name", type=str, default=None, help="The name of the experiment.")
    parser.add_argument("--model_save_path", type=str, default="models", help="The relative path to save model weights.")
    parser.add_argument("--steps_per_log", type=int, default=10, help="The number of steps to log loss and metrics.")
    parser.add_argument("--save_model_every", type=int, default=10, help="The number of steps to save model weights.")

    args = parser.parse_args()

    # Create a Config object from the args
    config_args = vars(args).copy()
    config_args['image_size'] = (config_args.pop('image_height'), config_args.pop('image_width')) 
    config = Config(**config_args)
    
    # Train the model
    trainer = Trainer(config)
    trainer.train()
