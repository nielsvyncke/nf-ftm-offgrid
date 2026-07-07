from typing import Optional, List, Dict, Any
from copy import deepcopy
import logging

import torch
from ignite.handlers import Checkpoint
from ignite.engine import Events


def binarize_label(x):
    """Binarize a label tensor: values > 0 become 1, else 0."""
    return (x > 0).to(x.dtype)


def identity(x):
    """Return the input unchanged."""
    return x


def disable_torch_weights_only():
    """Patch torch.load so weights_only defaults to False.

    PyTorch >= 2.6 changed the default to True, which breaks loading
    MONAI checkpoints that contain non-core globals such as
    MetricLoggerKeys. Call once before loading such checkpoints.
    """
    import torch

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load


def load_checkpoint(
    state_dict: Dict[str, Any],
    path: str,
    state_dict_options: Optional[Dict[str, Any]] = None,
    checkpoint_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    state_dict_options = state_dict_options or {}
    checkpoint_options = checkpoint_options or {}

    new_state_dict = deepcopy(state_dict)

    checkpoint_data = torch.load(path, **checkpoint_options)
    Checkpoint.load_objects(new_state_dict, checkpoint_data, **state_dict_options)

    return new_state_dict


def load_checkpoints(
    state_dict: Dict[str, Any], path_list: List[str], **kwargs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return [load_checkpoint(state_dict, path, **kwargs) for path in path_list]


class LogModelInfoHandler:
    def __init__(self, model):
        """
        Initializes the LogModelInfoHandler with the given model.

        Args:
            model (torch.nn.Module): The model to log information about.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

    def attach(self, engine):
        """
        Attaches the handler to the Ignite engine.

        Args:
            engine (ignite.engine.Engine): The Ignite engine to attach the handler to.
        """
        engine.add_event_handler(Events.STARTED, self.log_model_info)

    def log_model_info(self, engine):
        """
        Logs model information, such as total parameters, trainable parameters, and non-trainable parameters.

        Args:
            engine (ignite.engine.Engine): The Ignite engine.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params

        self.logger.info("==========================")
        self.logger.info(f"Model: {self.model._get_name()}")
        self.logger.info("==========================")
        self.logger.info(f"Total params (M): {total_params / 1e6:.1f}")
        self.logger.info(f"Trainable params (M): {trainable_params / 1e6:.1f}")
        self.logger.info(f"Non-trainable params: {non_trainable_params}")
        self.logger.info("==========================")
