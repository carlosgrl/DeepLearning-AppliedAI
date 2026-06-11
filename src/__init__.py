"""
LMC Project — Package init
"""

from .utils import set_seed, get_device, get_dataloaders, get_calibration_loader
from .models import MLP3, SimpleConvBN, ConvMixer, get_model, get_mobilenetv3
from .training import (
    train_one_epoch, evaluate, train_model, train_spawned_pair,
    run_multi_seed,
)

from .landscape import (
    compute_2d_loss_landscape,
    plot_4panel_landscape,
    plot_barrier_curves,
)
