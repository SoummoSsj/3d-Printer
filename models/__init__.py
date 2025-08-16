"""
SpeedNet Models Package
Contains neural network architectures for vehicle speed estimation
"""

from .speednet import (
    SpeedNet,
    SpeedNetLoss,
    CameraCalibrationModule,
    Vehicle3DModule,
    TemporalFusionModule,
    SpeedRegressionModule,
    VEHICLE_SIZE_PRIORS
)

__all__ = [
    'SpeedNet',
    'SpeedNetLoss', 
    'CameraCalibrationModule',
    'Vehicle3DModule',
    'TemporalFusionModule',
    'SpeedRegressionModule',
    'VEHICLE_SIZE_PRIORS'
]