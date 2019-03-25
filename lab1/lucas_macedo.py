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

import matplotlib.image as pic
import matplotlib.pyplot as plt
import numpy as np

# Capacidade
MINCAP = 0
MAXCAP = 256
# Intensidade
MINTENSITY = 0
MAXTENSITY = 255


# Q.02 OK
def imread(arquivo):
    imagem = pic.imread(arquivo).astype('uint8')
    return imagem


# Q.02a
# No PythonShell:
# import matplotlib.image as pic
# import matplotlib.pyplot as plt
# imagem = pic.imread("sua_imagem.sua_extensao")
# plt.imshow(imagem)
# plt.show()

# Q.02b


# Q.02c


# Q.03 OK
def nchannels(imagem):
    return 1 if len(imagem.shape) < 3 else 3


# Q.04 OK
def size(imagem):
    vetor = [0] * 2
    vetor[0] = imagem.shape[1]
    vetor[1] = imagem.shape[0]
    return vetor


# Q.05 OK
def rgb2gray(imagem):
    copia = imagem.copy()
    if (nchannels(imagem) == 3):
        vetor = size(imagem)
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                cinza = ((copia[x][y][0] * 0.299) + (copia[x][y][1] * 0.587) + (copia[x][y][2] * 0.114)) / 3
                copia[x][y][0] = cinza
                copia[x][y][1] = cinza
                copia[x][y][2] = cinza
        return copia
    else:
        return copia


# Q.06 OK
def imreadgray(arquivo):
    if nchannels(imread(arquivo)) < 3:
        return imread(arquivo)
    else:
        return rgb2gray(imread(arquivo))


# Q.07 OK
def imshow(imagem):
    if (nchannels(imagem) == 3):
        plt.imshow(imagem, interpolation='nearest')
    else:
        plt.imshow(imagem, cmap='gray', interpolation='nearest')
    plt.show()


# Q.08 OK
def thresh(imagem, limiar):
    copia = imagem.copy()
    vetor = size(copia)
    if (nchannels(copia) == 1):
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                copia[x][y] = MINTENSITY if (copia[x][y] < limiar) else MAXTENSITY
    else:
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                if (copia[x][y][0] > limiar):
                    copia[x][y][0] = MAXTENSITY
                else:
                    copia[x][y][0] = MINTENSITY

                if (copia[x][y][1] > limiar):
                    copia[x][y][1] = MAXTENSITY
                else:
                    copia[x][y][1] = MINTENSITY

                if (copia[x][y][2] > limiar):
                    copia[x][y][2] = MAXTENSITY
                else:
                    copia[x][y][2] = MINTENSITY
    return copia


# Q.09 OK
def negative(imagem):
    return MAXTENSITY - imagem


# Q.10 OK
def contrast(imagem, r, m):
    copia = imagem.copy()
    vetor = size(copia)
    if nchannels(copia) == 1:
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                value = (r * (copia[x][y] - m) + m)
                copia[x][y] = value
    else:
        for x in range(0, vetor[0]):
            for y in range(0, vetor[1]):
                value = (r * (copia[x][y][0] - m) + m)
                copia[x][y][0] = value
                value = (r * (copia[x][y][1] - m) + m)
                copia[x][y][1] = value
                value = (r * (copia[x][y][2] - m) + m)
                copia[x][y][2] = value
    return copia


# Q.10a OK
def imshow(imagem):
    if (nchannels(imagem) == 3):
        plt.imshow(imagem, interpolation='nearest')
    else:
        plt.imshow(imagem, interpolation='nearest')
    plt.show()


# Q.11 OK
def hist(imagem):
    try:
        histograma = np.zeros(shape=(3, MAXCAP))
        for red in np.nditer(imagem[:, :, 0]):
            histograma[0][red] += 1
        for green in np.nditer(imagem[:, :, 1]):
            histograma[1][green] += 1
        for blue in np.nditer(imagem[:, :, 2]):
            histograma[2][blue] += 1
    except IndexError:
        histograma = np.zeros(shape=(MAXCAP))
        for gray in np.nditer(imagem):
            histograma[gray] += 1
    return histograma


# Q.12/13
def showhist(histograma, bin=1):
    length = int(calchistlength(histograma, bin))
    xvalues = np.arange(length) * bin
    if bin != 1:
        for x in range(0, length - 1):
            xvalues[x] = xvalues[x + 1] - 1
        xvalues[length - 1] = MAXTENSITY
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    if isgrayhist(histograma):
        groupvector = grouphist(histograma, bin, length)
        greyrect = ax.bar(xvalues, groupvector, width, color='w')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade em escala de cinza')
    else:
        groupvectorred = grouphist(histograma[0], bin, length)
        groupvectorgreen = grouphist(histograma[1], bin, length)
        groupvectorblue = grouphist(histograma[2], bin, length)
        redrect = ax.bar(xvalues, groupvectorred, width, color='r')
        greenrect = ax.bar(xvalues + width, groupvectorgreen, width, color='g')
        bluerect = ax.bar(xvalues + width * 2, groupvectorblue, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade')
        ax.legend((redrect[0], greenrect[0], bluerect[0]), ('RED', 'GREEN', 'BLUE'))
    # Configuracoes visuais:
    if bin >= 5:
        ax.set_xticks(xvalues)
    if bin > 1:
        ax.set_xlabel('Pixels agrupados por ' + str(bin))
    plt.show()
    return


# Q.14
def histeq(imagem):
    imgequalized = imagem.copy()
    imgsize = size(imagem)
    qtdpixels = imgsize[0] * imgsize[1]
    histogram = hist(imagem)
    histqualized = pmf(histogram, qtdpixels)
    cdf(histqualized)
    if isgrayhist(histqualized):
        for x in range(0, MAXCAP):
            histqualized[x] *= MAXTENSITY
    else:
        for x in range(0, 256):
            histqualized[0][x] *= MAXTENSITY
            histqualized[1][x] *= MAXTENSITY
            histqualized[2][x] *= MAXTENSITY
    # Aplica o resultado da equalizacao na imagem
    if isgrayhist(histqualized):
        for x in range(0, imgsize[0]):
            for y in range(0, imgsize[1]):
                imgequalized[x][y] = int(histqualized[imgequalized[x][y]])
    else:
        for x in range(0, imgsize[0]):
            for y in range(0, imgsize[1]):
                imgequalized[x][y][0] = int(histqualized[0][imgequalized[x][y][0]])
                imgequalized[x][y][1] = int(histqualized[1][imgequalized[x][y][1]])
                imgequalized[x][y][2] = int(histqualized[2][imgequalized[x][y][2]])
    return imgequalized


# Q.15
def convolve(imagem, mascara):
    if type(mascara) is np.ndarray:
        mascara = mascara.tolist()  # converte para lista casa nao seja.
    vetor = size(imagem)
    convolucao = imagem.copy()
    for i in range(0, vetor[0]):
        for j in range(0, vetor[1]):
            if nchannels(imagem) < 3:
                result = calcconvolve(i, j, imagem, mascara)
                result = truncate(result)
                convolucao[i][j] = result
            else:
                resultRGB = calcconvolve(i, j, imagem, mascara)
                convolucao[i][j][0] = truncate(resultRGB[0])
                convolucao[i][j][1] = truncate(resultRGB[1])
                convolucao[i][j][2] = truncate(resultRGB[2])
    return convolucao


# Q.16
def maskblur():
    return [[0.0625, 0.1250, 0.0625], [0.125, 0.2500, 0.1250], [0.0625, 0.1250, 0.0625]]


# Q.17
def blur(imagem):
    return convolve(imagem, maskblur())


# Q.18
def seSquare3():
    return [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


# Q.19
def seCross3():
    return [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


# Q.20
def erode(imagem, estruturante):
    return erosion(imagem, estruturante, MAXTENSITY, min)


def dilate(imagem, estruturante):
    return dilation(imagem, estruturante, MINTENSITY, max)


##################################################
##########Funções Auxiliares######################
##################################################

def isgrayhist(histograma):
    return histograma.shape[0] == 1


# Define o centro da mascara.
def center(mask):
    length = masklen(mask)
    return int(length[0] / 2), int(length[1] / 2)


# retorna uma tupla com a quantidade de linhas e colunas da mascara.
def masklen(mask):
    return len(mask), len(mask[0])


# Coloca na posicao inicial da mascara de acordo com a imagem.
def relativeposition(posx, posy, centerposition):
    posx -= centerposition[0]
    posy -= centerposition[1]
    return posx, posy


def outofbounds(positionx, positiony, image, ):
    imagelength = size(image)
    if positionx < 0:
        positionx = 0
    elif positionx > (imagelength[0] - 1):
        positionx = imagelength[0] - 1

    if positiony < 0:
        positiony = 0
    elif positiony > (imagelength[1] - 1):
        positiony = imagelength[1] - 1

    return positionx, positiony


# Calcula o valor do pixel da imagem multiplicado pelo peso.
def calcpixel(positionx, positiony, weight, image):
    # Verifica se alguam coordenada esta fora da imagem e ajusta para o correto.
    position = outofbounds(positionx, positiony, image)
    if nchannels(image) < 3:
        return (image[position[0]][position[1]]) * weight
    else:
        # RGB
        r = image[position[0]][position[1]][0]
        g = image[position[0]][position[1]][1]
        b = image[position[0]][position[1]][2]

        return r * weight, g * weight, b * weight


def truncate(value):
    v = int(value)
    if v > MAXTENSITY:
        return MAXTENSITY
    elif v < MINTENSITY:
        return MINTENSITY

    return v


def calcconvolve(positionx, positiony, imagem, mascara):
    centro = center(mascara)
    mascaralength = masklen(mascara)
    mascarapos = relativeposition(positionx, positiony, centro)
    result = 0.0
    resultr = 0.0
    resultg = 0.0
    resultb = 0.0
    if nchannels(imagem) < 3:
        for i in range(0, mascaralength[0]):
            for j in range(0, mascaralength[1]):
                x = mascarapos[0] + i
                y = mascarapos[1] + j
                result += calcpixel(x, y, mascara[i][j], imagem)
        return result
    else:
        for i in range(0, mascaralength[0]):
            for j in range(0, mascaralength[1]):
                x = mascarapos[0] + i
                y = mascarapos[1] + j
                valuergb = calcpixel(x, y, mascara[i][j], imagem)
                resultr += valuergb[0]
                resultg += valuergb[1]
                resultb += valuergb[2]
        return resultr, resultg, resultb


# Calcula o tamanho de posicoes necessarias com base no valor de agrupamento 'bin'.
def calchistlength(hist, bin):
    if isgrayhist(hist):
        length = (len(hist) / bin)
        if len(hist) % bin != 0:
            length += 1
    else:
        length = (len(hist[0]) / bin)
        if len(hist[0]) % bin != 0:
            length += 1
    return length


# Agrupa os valores do vetor de acordo com o valor bin.
def grouphist(hist, bin, length):
    groupvector = [0] * length
    count = 0
    acum = 0
    position = 0
    for x in range(0, 256):
        if count <= bin:
            acum += hist[x]
        count += 1
        if count > (bin - 1):
            groupvector[position] = acum
            acum = 0
            count = 0
            position += 1

    if count != 0:
        groupvector[position] = acum
    return groupvector


def pmf(hist, qtdpixels):
    if isgrayhist(hist):
        histequ = [float(i) for i in hist]
        for x in range(0, 256):
            histequ[x] /= qtdpixels
        return histequ
    else:
        histequ = [0] * 3
        histequ[0] = [float(i) for i in hist[0]]
        histequ[1] = [float(i) for i in hist[1]]
        histequ[2] = [float(i) for i in hist[2]]
        for x in range(0, 256):
            histequ[0][x] /= qtdpixels
            histequ[1][x] /= qtdpixels
            histequ[2][x] /= qtdpixels
        return histequ


# Acumula os valores do histograma.
def cdf(hist):
    if isgrayhist(hist):
        for x in range(1, 256):
            hist[x] = hist[x] + hist[x - 1]
    else:
        for x in range(1, 256):
            hist[0][x] = hist[0][x] + hist[0][x - 1]
            hist[1][x] = hist[1][x] + hist[1][x - 1]
            hist[2][x] = hist[2][x] + hist[2][x - 1]

    return hist


def erosion(imagem, estruturante, centrox, centroy):
    aux = np.shape(imagem)

    if np.size(aux) > 2:
        imagem = imagem[:, :, 0]
        aux = np.shape(imagem)

    erosao = []
    erosaoline = []
    tam = np.shape(estruturante)
    check = 0  # flag para indicar se o estruturante está completamente contido no objeto
    total = 0  # flag para indicar a quantidade de pixels 1 dentro do elemento estruturante

    for u in range(tam[0]):
        for v in range(tam[1]):
            if estruturante[u][v] != 0:
                total += 1

    for x in range(aux[0]):
        for y in range(aux[1]):
            if imagem[x][y] == 1:  # verifica se o pixel sobre analise é branco
                for u in range(0 - centrox, tam[0] - centrox):
                    for v in range(0 - centroy, tam[1] - centroy):
                        if x + u >= 0 and x + u < aux[0] and y + v >= 0 and y + v < aux[1]:
                            check += estruturante[u + centrox][v + centroy] * imagem[x + u][y + v]
                if check == total:
                    erosaoline.append(1)  # adiciona-se um pixel igual a 1
                else:  # caso contrário, adiciona-se um pixel igual a 0
                    erosaoline.append(0)
                check = 0
            else:
                erosaoline.append(0)
        erosao.append(erosaoline)
        erosaoline = []

    return erosao


def dilation(imagem, estruturante, centrox, centroy):
    aux = np.shape(imagem)

    if np.size(aux) > 2:
        imagem = imagem[:, :, 0]  # seleciona apenas uma matriz de cor caso a leitura seja rgb
        aux = np.shape(imagem)

    for x in range(aux[0]):  # cria-se o complemento da imagem
        for y in range(aux[1]):
            imagem[x][y] = 1 - imagem[x][y]

    dilatacao = erosion(imagem, estruturante, centrox, centroy)  # erode o complemento da imagem

    for x in range(aux[0]):  # tira-se o complemento da imagem erodida
        for y in range(aux[1]):
            dilatacao[x][y] = 1 - dilatacao[x][y]

    # corrige os valores das bordas

    dilatacao[:][-1] = dilatacao[:][-2]
    dilatacao[:][0] = dilatacao[:][1]
    dilatacao[0][:] = dilatacao[1][:]
    dilatacao[-1][:] = dilatacao[-2][:]

    return dilatacao
