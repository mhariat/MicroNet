from nn.quantization.quantnet import QuantizedNet
from nn.quantization.uniform import Uniform
from nn.quantization.wage import Wage, WageSGD, WageFunction
from nn.quantization.agile import Agile, AgileFunction, AgileSGD
from nn.quantization.sigmamax import SigmaMax, SigmaMaxQuantFunction
from nn.quantization.max import Max, MaxQuantFunction

__all__ = [
        'QuantizedNet',
        'Uniform', 
        'Wage',
        'WageFunction',
        'WageSGD',
        'Agile',
        'AgileFunction',
        'AgileSGD',
        'SigmaMax',
        'SigmaMaxQuantFunction',
        'Max',
        'MaxQuantFunction',
        ]
