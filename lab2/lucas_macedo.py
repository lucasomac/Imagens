import matplotlib.image as pic
import numpy as np

from lab1 import lucas_macedo as lc


# Questao 01
def componente_conexo(imagem, n):
    saida = lc.thresh(imagem, 128)
    rotulo_atual = 1
    coords = []
    for i in range(saida.shape[0]):
        for j in range(saida.shape[1]):
            if saida[i][j] == 255:
                saida[i][j] = rotulo_atual
                coords.append((i, j))
                while coords.__len__() != 0:
                    cord_atual = coords.pop()
                    for p in adj(n):
                        if saida[cord_atual[0] + p[0]][cord_atual[1] + p[1]] == 255:
                            saida[cord_atual[0] + p[0]][cord_atual[1] + p[1]] = rotulo_atual
                            coords.append((cord_atual[0] + p[0], cord_atual[1] + p[1]))
                rotulo_atual += 1
    return np.asarray(saida, np.uint16)


# Questao 02

def rgb_pseudo(imagem):
    saida = imagem.copy()
    saida = saida * 3
    return saida


# Funcoes Auxiliares
def imread16(arquivo):
    imagem = pic.imread(arquivo).astype('uint16')
    return imagem


def adj(inteiro):
    if (inteiro == 4):
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        return ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
