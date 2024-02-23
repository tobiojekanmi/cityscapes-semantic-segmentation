import os
import json
import torch
from datetime import datetime
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from modules.data.datamanager import CityScapesDataManagerConfig, CityScapesDataManager
from modules.models import Model, ModelConfig
from modules.evaluate import Evaluator
from modules.configs import Config


class Trainer(torch.nn.Module):

    """
    The Trainer class to train a model on a dataset.
    """

    def __init__(self, config: Config):
        """
        Instantiates an object of the Trainer class.

        Args:
            config (Config): The configuration object.
        """
        super(Trainer, self).__init__()
        self.config = config
        self._init_trainer_modules(config)

    def train_step(self, model, dataloader, criterion, optimizer, device):
        """
        Trains a model for a single epoch.
        """
        model = model.to(device)
        model.train()
        running_loss = 0.
        running_examples = 0
        
        for _, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            running_loss += loss
            running_examples += len(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        return running_loss, running_examples
    
    def evaluate_step(self, model, dataloader, criterion, device):
        """
        Evaluates a model for a single epoch.
        """
        model = model.to(device)
        model.eval()
        running_loss = 0.
        running_examples = 0
        
        with torch.inference_mode():
            for _, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                running_loss += loss
                running_examples += len(images)

        torch.cuda.empty_cache()
        return running_loss, running_examples

    def train(self):
        """
        Trains a model for a single epoch.
        """
        writer = SummaryWriter(self.log_path)

        for epoch in range(self.config.num_epochs):
            train_loss, train_examples = self.train_step(
                self.model, 
                self.train_dataloader, 
                self.criterion, 
                self.optimizer, 
                self.device,
                )

            if (
                epoch == 0 or epoch+1 % self.config.steps_per_log == 0 or 
                epoch+1 % self.config.evaluate_model_every  == 0
                ):
                valid_loss, valid_examples = self.evaluate_step(
                    self.model, 
                    self.valid_dataloader, 
                    self.criterion, 
                    self.device,
                    )
            
            if epoch == 0 or epoch+1 % self.config.steps_per_log == 0:
                print(
                    f'Epoch: [{epoch+1}/{self.config.num_epochs}], \
                        lr: {self.optimizer.param_groups[0]["lr"] :.2}, \
                        Train loss: {train_loss/train_examples:.2}, \
                        Valid loss: {valid_loss/valid_examples:.2}'
                    )

            if epoch == 0 or epoch+1 % self.config.save_model_every == 0:
                model_save_path = os.path.join(
                    self.model_save_path, f'model_{epoch+1}.pth'
                    )
                torch.save(self.model.state_dict(), model_save_path)
            
            if epoch == 0 or epoch+1 % self.config.evaluate_model_every:
                train_metrics = self.train_evaluator(
                    self.model, self.train_dataloader
                    )
                valid_metrics = self.valid_evaluator(
                    self.model, self.valid_dataloader
                    )

                writer.add_scalar('Learning rate', self.optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar('Loss/train', train_loss/train_examples, epoch)
                writer.add_scalar('Loss/valid', valid_loss/valid_examples, epoch)

                for metric_name, metric_value in train_metrics["metrics"].items():
                    writer.add_scalar(f'Average Metrics/train/{metric_name}', metric_value.mean().cpu().item(), epoch)
                for metric_name, metric_value in valid_metrics["metrics"].items():
                    writer.add_scalar(f'Average Metrics/valid/{metric_name}', metric_value.mean().cpu().item(), epoch)

                if self.config.eval_individual_class_metrics:
                    for metric_name, metric_value in train_metrics["class_metrics"].items():
                        writer.add_scalar(f'Class Metrics/train/{metric_name}', metric_value.cpu(), epoch)
                    for metric_name, metric_value in valid_metrics["class_metrics"].items():
                        writer.add_scalar(f'Class Metrics/valid/{metric_name}', metric_value.cpu(), epoch)


            if self.config.use_scheduler:
                self.scheduler.step()

    
    def _init_trainer_modules(self, config):
        self.device = torch.device(self.config.device_type)

        # Initialize a Model
        self.model = Model(
            ModelConfig(
                model_name = config.model_name,
                encoder_name = config.encoder_name,
                in_channels = config.in_channels,
                encoder_depth = config.encoder_depth,
                encoder_weights = config.encoder_weights,
                decoder_channels = config.decoder_channels,
                num_classes = config.num_classes,
                activation = config.activation,
                weights_path = config.weights_path,
                device = self.device
                )
            )

        # Initialize Train and Valid Dataloaders
        self.datamanager = CityScapesDataManager(
            CityScapesDataManagerConfig(
                images_dir = config.images_dir,
                labels_dir = config.labels_dir,
                image_size = config.image_size,
                ignore_void_classes= config.ignore_void_classes,
                ignore_index = config.ignore_index,
                ignore_labels= config.ignore_labels,
                train_transforms = config.train_transforms,
                valid_transforms = config.valid_transforms,
                test_transforms = config.test_transforms,
                batch_size = config.batch_size,
                num_workers = config.num_workers,
                pin_memory = config.pin_memory,
                shuffle = config.shuffle,
                collate_fn = config.collate_fn,
                drop_last = config.drop_last
                )
            )
        self.train_dataloader = self.datamanager.get_train_dataloader()
        self.valid_dataloader = self.datamanager.get_valid_dataloader()
        
        # Initialize a Loss, Optimizer, and Scheduler
        if self.config.use_cross_entropy_loss:
            self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index = self.datamanager.ignore_index
                )
        else:
            self.criterion = smp.losses.DiceLoss(
                "multiclass", 
                log_loss=False, 
                from_logits=True, 
                ignore_index = self.datamanager.ignore_index
                )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = config.learning_rate,
            weight_decay = config.weight_decay
            )
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size = config.scheduler_step_size,
                gamma = config.scheduler_gamma
                )
        
        # Initialize the Evalution Metrics
        self.train_evaluator = Evaluator(
            n_classes = config.num_classes,
            ignore_index= self.datamanager.ignore_index,
            eval_individual_class_metrics = self.config.eval_individual_class_metrics,
            device = self.device
        )
        self.valid_evaluator = Evaluator(
            n_classes = config.num_classes,
            ignore_index= self.datamanager.ignore_index,
            eval_individual_class_metrics = self.config.eval_individual_class_metrics,
            device = self.device
        )

        # Initialize the Logging details
        self.output_path = os.path.join(
            config.output_dir,
            config.experiment_name if not None else '',
            datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            )
        self.log_path = os.path.join(self.output_path, config.log_path)
        self.model_save_path = os.path.join(self.output_path, config.model_save_path)
        
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.log_path):
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.model_save_path):
            Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        
        with open(os.path.join(self.output_path, "config.json"), 'w') as config_file:
            json.dump(vars(config), config_file)
