from collections import deque as dq
import matplotlib.image as pic


def imread(arquivo):
    imagem = pic.imread(arquivo).astype('uint16')
    return imagem


def componente_conexo(imagem, n):
    copia = imagem.copy()
    rotulo_atual = 1
    coords = dq.deque()
    for i in range(copia.shape[0]):
        for j in range(copia.shape[1]):
            if (copia[i][j] == 255):
                copia[i][j] = rotulo_atual
                coords.append((i, j))
                while (coords.isEmpty() != True):
                    coords.popleft()
                for p in range(adj(n)):
                    if copia[i + p[0]][j + p[1]] == 255:
                        copia[i + p[0]][j + p[1]] = rotulo_atual
                        coords.append()
            rotulo_atual += 1


def adj(inteiro):
    if (inteiro == 4):
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        return ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
