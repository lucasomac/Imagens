import matplotlib.image as pic
import matplotlib.pyplot as plt
import numpy as np


def imread(arquivo):
    imagem = pic.imread(arquivo).astype('uint8')
    return imagem


def negative(imagem):
    return 255 - imagem


def convert_float_uint(imagem):
    return (imagem * 255).astype(np.uint8)


def clamp(value, maior, menor):
    return max(min(value, maior), menor)
