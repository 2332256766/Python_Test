import numpy as np
import matplotlib.pyplot as plt
from .plotDataPoints import plotDataPoints

def drawLine(p1, p2, *varargin):
    plt.plot([p1[0],p2[0]], [p1[1], p2[1]], *varargin)

