import os
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassF1Score, 
    MulticlassJaccardIndex,
    MulticlassConfusionMatrix
    )
from modules.models import Model, ModelConfig
from modules.data.datamanager import CityScapesDataManager, CityScapesDataManagerConfig

class Evaluator(torch.nn.Module):
    """
    The Evaluator class to evaluate models on a dataset.
    """

    def __init__(
            self, 
            n_classes: int, 
            average: Optional[str] = 'macro', 
            ignore_index: Optional[int] = None, 
            eval_individual_class_metrics: bool = False,
            device: Optional[Union[str, torch.device]] = 'cpu'
            ):
        """
        Instantiates an object of the Evaluator class.

        Args:
            n_classes (int): Number of classes in the dataset.
            average (Optional[str], optional): The type of averaging to use. 
                One of ('micro', 'macro', 'weighted'). Defaults to 'macro'.
            ignore_index (Optional[int], optional): The index to ignore. 
                Defaults to None.
            eval_individual_class_metrics (bool, optional): Whether to evaluate
                individual class metrics. Defaults to False.
        """
        super(Evaluator, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.eval_individual_class_metrics = eval_individual_class_metrics
        self.metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=n_classes, 
                    average=average, 
                    ignore_index=ignore_index
                    ),
                MulticlassF1Score(
                    num_classes=n_classes, 
                    average=average, 
                    ignore_index=ignore_index
                    ),
                MulticlassJaccardIndex(
                    num_classes=n_classes, 
                    average=average, 
                    ignore_index=ignore_index
                    ),
                MulticlassConfusionMatrix(
                    num_classes=n_classes, 
                    normalize='true', 
                    ignore_index=ignore_index
                    )
            ]
        ).to(self.device)
 
        if self.eval_individual_class_metrics:
            self.class_metrics = MetricCollection(
                [
                    MulticlassAccuracy(
                        num_classes=n_classes, 
                        average=None, 
                        ignore_index=ignore_index
                        ),
                    MulticlassF1Score(
                        num_classes=n_classes, 
                        average=None, 
                        ignore_index=ignore_index
                        ),
                    MulticlassJaccardIndex(
                        num_classes=n_classes, 
                        average=None, 
                        ignore_index=ignore_index
                        ),
                    MulticlassConfusionMatrix(
                        num_classes=n_classes, 
                        normalize='true', 
                        ignore_index=ignore_index
                        )
                ]
            ).to(self.device)
    
    def reset(self):
        """
        Resets all evaluation metrics.
        """
        self.metrics.reset()
        if self.eval_individual_class_metrics:
            self.class_metrics.reset()


    def update(self, predictions, targets):
        """
        Updates evaluation metrics with predictions and ground truth targets.

        Args:
            predictions (torch.Tensor): Predicted segmentation maps (N x H x W).
            targets (torch.Tensor): Ground truth segmentation maps (N x H x W).
        """
        self.metrics.update(predictions, targets)
        if self.eval_individual_class_metrics:
            self.class_metrics.update(predictions, targets)

    def forward(self, model, dataloader):
        """
        Evaluates a model on a dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
        """
        self.reset()
        model = model.to(self.device)
        model.eval()

        with torch.inference_mode():
            for _, (images, targets) in enumerate(dataloader):
                predictions = model(images.to(self.device))
                self.update(predictions, targets.to(self.device))

        metrics = self.metrics.compute()
        class_metrics = None
        if self.eval_individual_class_metrics:
            class_metrics = self.class_metrics.compute()

        return {
            "metrics": metrics, "class_metrics": class_metrics}
            

def metrics_collate_fn(list_of_dicts):
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists



if __name__ == "__main__":
    datamanager_config = CityScapesDataManagerConfig(
        images_dir = "datasets/images/leftImg8bit/", 
        labels_dir = "datasets/labels/gtFine", 
        image_size = (256, 256),
        batch_size = 2,
        shuffle = True
    )
    datamanager = CityScapesDataManager(datamanager_config)    
    train_dataloader = datamanager.get_train_dataloader()
    train_batch = next(iter(train_dataloader))

    print("Loading Model")
    model = Model(ModelConfig(model_name='UnetPlusPlus', num_classes=19))
    
    print("Predicting with Model")
    predictions = model(train_batch[0])

    print("Evaluating Model")
    evaluator = Evaluator(n_classes=19, eval_individual_class_metrics=True, device='cuda')
    metrics = evaluator(model, train_dataloader)
    print(metrics['metrics'].keys())
    print(metrics['metrics']['MulticlassAccuracy'].cpu().item())
