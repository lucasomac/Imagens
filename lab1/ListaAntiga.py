########################################
#
# Nome: Lucas de Oliveira Macedo
# Matricula:201500018252
# Email:lucasomac@outlook.com
#
# Nome:
# Matricula:
# Email:
#
########################################

# import matplotlib as mat
import matplotlib.image as pic
import matplotlib.pyplot as plt

# from PIL import Image

# Capacidade
MINCAP = 0
MAXCAP = 256
# Intensidade
MINTENSITY = 0
MAXTENSITY = 255


# Q.02 OK
# def imread(arquivo):
#     imagem = pic.open(arquivo)
#     return np.asarray(imagem, "uint8")
# Q.02 OK
def imread(arquivo):
    imagem = pic.imread(arquivo)
    return imagem


# Q.02a
# No PythonShell:
# from PIL import Image
# imagem = Image.open("sua_imagem.sua_extensao")
# imagem.show()

# Q.02b
# No PythonShell:
# from PIL import Image
# imagem = Image.open("sua_imagem.sua_extensao").convert('L')
# img.show()

# Q.02c
# No PythonShell:
# from PIL import Image
# imagem = Image.open("sua_imagem.sua_extensao")
# imagem.resize((50,50)).show()

# Q.03 OK
def nchannels(imagem):
    return 1 if len(imagem.shape) < 3 else 3


# Q.04 OK
def size(imagem):
    vetor = [0] * 2
    vetor[0] = imagem.shape[0]
    vetor[1] = imagem.shape[1]
    return vetor


# Q.05 OK
def rgb2gray(imagem):
    copia = imagem.copy()
    vetor = size(imagem)
    for x in range(0, vetor[0]):
        for y in range(0, vetor[1]):
            cinza = ((copia[x][y][0] * 0.299) + (copia[x][y][1] * 0.587) + (copia[x][y][2] * 0.114)) / 3
            copia[x][y][0] = cinza
            copia[x][y][1] = cinza
            copia[x][y][2] = cinza
    return copia


# Q.06 OK
def imreadgray(arquivo):
    if nchannels(imready(arquivo)) < 3:
        return imready(arquivo)
    else:
        return rgb2gray(imready(arquivo))


# Q.07 OK
def imshow(imagem):
    if (nchannels(imagem) == 3):
        plt.imshow(img, interpolation='nearest')
    else:
        plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()


# Q.08
def thresh(imagem, limiar):
    imagem2 = imagem.copy()
    vetor = size(imagem2)
    if (nchannels(imagem2) == 1):
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                imagem2[x][y] = MINTENSITY if (imagem2[x][y] < limiar) else MAXTENSITY
    else:
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                imagem2[x][y][0] = MINTENSITY if (imagem2[x][y] < limiar) else MAXTENSITY
                imagem2[x][y][1] = MINTENSITY if (imagem2[x][y] < limiar) else MAXTENSITY
                imagem2[x][y][2] = MINTENSITY if (imagem2[x][y] < limiar) else MAXTENSITY
    return imagem2
