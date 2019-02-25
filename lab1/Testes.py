import matplotlib.image as npimg
import matplotlib.pyplot as plt
import numpy as np


# _____________Funcao imread (segunda questao)
def imread(imagem):  # inserindo imagem por parametro
    img = npimg.imread(imagem)  # Convertendo a imagem para ndarray
    return img  # Verificando conversao da imagem


# _____________Funcao nchannels (terceira questao)
def nchannels(img):  # inserindo imagem por parametro
    # verificando se a imagem tem mais de uma cor
    try:
        return len(img[0][0])
    except:
        return 1


# _____________Funcao size (quarta questao)
def size(img):  # inserindo imagem por parametro
    size = [len(img[0]),
            len(img)]  # criando um array size com a primeira posicao sendo a largura e a segunda sendo a altura
    return size


# _____________Funcao rgb2gray (quinta questao)
def rgb2gray(img):
    newImg = img.copy()  # Hardcopy

    if (nchannels(img) == 3):  # Se for colorida, converte e retorna a imagem convertida
        gray = np.dot(newImg[..., :3],
                      [0.299, 0.587, 0.144])  # Multiplicacao de matrizes: a matriz da imagem pelo vetor de pesos.
        return gray

    else:  # Se nao for, retorna a imagem
        return newImg


# _____________Funcao imreadgray (sexta questao)
def imreadgray(img):
    if (nchannels(img) == 3):
        return rgb2gray(img)

    else:
        return img


# _____________Funcao imshow (setima questao)
def imshow(img):
    newImg = img.copy()  # Hardcopy da imagem

    if (nchannels(img) != 3):
        # Se a imagem for cinza
        plt.imshow(newImg, cmap=plt.get_cmap('gray'), interpolation="nearest")
        plt.show()

    else:
        image = plt.imshow(newImg, interpolation="nearest")
        plt.show()


def imshow2(img):
    newImg = img.copy()  # Hardcopy da imagem

    if (nchannels(img) != 3):
        # Se a imagem for cinza
        plt.imshow(newImg, interpolation="nearest")
        plt.show()

    else:
        image = plt.imshow(newImg, interpolation="nearest")
        plt.show()


# _____________Funcao thresh (oitava questao)
def thresh(img, lim):
    newImg = img.copy()
    if (nchannels(img) == 1):  # verificando imagem em escala de cinza
        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                if (newImg[x][y] >= lim):  # Verificando limiar
                    newImg[x][y] = 255
                else:
                    newImg[x][y] = 0

    else:  # verificando imagem RGB

        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                if (newImg[x][y][0] >= lim):  # Verificando limiar
                    newImg[x][y][0] = 255
                else:
                    newImg[x][y][0] = 0
                if (newImg[x][y][1] >= lim):
                    newImg[x][y][1] = 255
                else:
                    newImg[x][y][1] = 0
                if (newImg[x][y][2] >= lim):
                    newImg[x][y][2] = 255
                else:
                    newImg[x][y][2] = 0
    return newImg


# _____________Funcao negative (nona questao)
def negative(img):
    newImg = img.copy()

    if (nchannels(img) == 1):  # verificando imagem em escala de cinza

        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                newImg[x][y] = 255 - newImg[x][y]  # Invertendo a imagem

    else:  # verificando imagem RGB

        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                newImg[x][y][0] = 255 - newImg[x][y][0]  # Invertendo a imagem
                newImg[x][y][1] = 255 - newImg[x][y][1]
                newImg[x][y][2] = 255 - newImg[x][y][2]

    return newImg


# _____________Funcao contrast (decima questao)
def contrast(img, r, m):
    newImg = img.copy()
    if (nchannels(img) == 1):  # verificando imagem em escala de cinza
        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                tmp = r * (img[x][y] - m) + m
                if (tmp >= 255):
                    newImg[x][y] = 255
                elif (tmp <= 0):
                    newImg[x][y] = 0
    else:  # verificando imagem RGB
        for x in range(0, len(newImg)):  # iterando array em largura e altura
            for y in range(0, len(newImg[0])):
                tmp = r * (img[x][y][0] - m) + m
                if (tmp >= 255):
                    newImg[x][y][0] = 255
                elif (tmp <= 0):
                    newImg[x][y][0] = 0
                tmp = r * (img[x][y][1] - m) + m
                if (tmp >= 255):
                    newImg[x][y][1] = 255
                elif (tmp <= 0):
                    newImg[x][y][1] = 0
                    tmp = r * (img[x][y][2] - m) + m
                if (tmp >= 255):
                    newImg[x][y][2] = 255
                elif (tmp <= 0):
                    newImg[x][y][2] = 0
    return newImg


# _______________Funcao hist (Decia primeira questao)
def hist(img):
    tamanho = size(img)

    if (nchannels(img) == 3):  # Imagem colorida
        resultado_colorido = [[0 for x in range(3)] for y in range(256)]  # Cria uma matriz de 3 coluna e 256 linhas

        for x in xrange(0, len(img)):  # Eixo x
            for y in xrange(0, len(img[0])):  # Eixo y
                resultado_colorido[img[x][y][0]][0] = (resultado_colorido[img[x][y][0]][0]) + 1  # R
                resultado_colorido[img[x][y][1]][1] = (resultado_colorido[img[x][y][1]][1]) + 1  # G
                resultado_colorido[img[x][y][2]][2] = (resultado_colorido[img[x][y][2]][2]) + 1  # B

        return resultado_colorido

    else:  # Imagem cinza
        resultado_cinza = [0 for x in range(256)]  # Cria um vetor de 256 posicoes

        for x in xrange(0, tamanho[1]):
            for y in xrange(0, tamanho[0]):
                resultado_cinza[img[x][y]] = (resultado_cinza[img[x][y]]) + 1

        return resultado_cinza


# _______________Funcao showhist( Decima segunda questao)
def showhist(imagem):
    try:
        red_pixels = [0 for x in range(256)]
        green_pixels = [0 for x in range(256)]
        blue_pixels = [0 for x in range(256)]

        for x in xrange(0, 256):
            red_pixels[x] = imagem[x][0]
            green_pixels[x] = imagem[x][1]
            blue_pixels[x] = imagem[x][2]

        fig, ax = plt.subplots()
        index = np.arange(256)
        bar_width = 0.10
        opacity = 0.8

        red = plt.bar(index, red_pixels, bar_width,
                      alpha=opacity,
                      color='r',
                      label='Red')

        green = plt.bar(index + bar_width, green_pixels, bar_width,
                        alpha=opacity,
                        color='g',
                        label='Green')

        blue = plt.bar(index + (bar_width * 2), blue_pixels, bar_width,
                       alpha=opacity,
                       color='b',
                       label='Blue')

        plt.xlabel('Intensidade')
        plt.ylabel('Frequencia')
        plt.title('Histograma')
        plt.xticks(index + bar_width, range(0, 256))
        plt.legend()

        plt.tight_layout()
        plt.show()

    except:
        gray_pixels = [0 for x in range(256)]
        for x in range(0, 256):
            gray_pixels[x] = imagem[x]

        fig, ax = plt.subplots()
        index = np.arange(256)
        bar_width = 0.10
        opacity = 0.8

        gray = plt.bar(index, gray_pixels, bar_width,
                       alpha=opacity,
                       color='g',
                       label='Gray')

        plt.xlabel('Intensidade')
        plt.ylabel('Frequencia')
        plt.title('Histograma')
        plt.xticks(index, range(0, 256))
        plt.legend()

        plt.tight_layout()
        plt.show()


# _______________Funcao showhist2 ( Decima terceira questao)
def showhist2(imagem, bin):
    try:
        red_pixels = [0 for x in range(256 + 1 / bin)]
        green_pixels = [0 for x in range(256 + 1 / bin)]
        blue_pixels = [0 for x in range(256 + 1 / bin)]
        counter = 0

        for x in xrange(0, 256):
            if (x % bin == 0 and x != 0):
                counter += 1

            red_pixels[counter] += imagem[x][0]
            green_pixels[counter] += imagem[x][1]
            blue_pixels[counter] += imagem[x][2]

        fig, ax = plt.subplots()
        index = np.arange(256 + 1 / bin)
        bar_width = 0.15
        opacity = 0.8

        red = plt.bar(index, red_pixels, bar_width,
                      alpha=opacity,
                      color='r',
                      label='Red')

        green = plt.bar(index + bar_width, green_pixels, bar_width,
                        alpha=opacity,
                        color='g',
                        label='Green')

        blue = plt.bar(index + (bar_width * 2), blue_pixels, bar_width,
                       alpha=opacity,
                       color='b',
                       label='Blue')

        plt.xlabel('Intensidade')
        plt.ylabel('Frequencia')
        plt.title('Histograma')
        plt.xticks(index + bar_width, range(0, 256 / bin))
        plt.legend()

        plt.tight_layout()
        plt.show()

    except:

        gray_pixels = [0 for x in range(0, 256 / bin)]
        counter = 0

        for x in range(0, 256):
            if (x % bin == 0 and x != 0):
                counter += 1
            gray_pixels[counter] += imagem[x]

        fig, ax = plt.subplots()
        index = np.arange(256 / bin)
        bar_width = 0.102
        opacity = 0.8

        gray = plt.bar(index, gray_pixels, bar_width,
                       alpha=opacity,
                       color='g',
                       label='Gray')

        plt.xlabel('Intensidade')
        plt.ylabel('Frequencia')
        plt.title('Histograma')
        plt.xticks(index, range(0, 256 / bin))
        plt.legend()

        plt.tight_layout()
        plt.show()


'''
Observacao quanto a essa questao e que apesar de quando coloca para imprimir pelo comando print, o resultado mostrado e diferente 
do resultado quando se acessa as posicoes finais do array, por exemplo no caso de scarletG, o valor 253 nao aparece no print se nao me engano,
mas no valor 254 a soma das probabilidaddes e 1, chegando a conclusao de que esta certo(Nao tem nenhum valor com intensidade 256, logo a probabilidade dele acontecer e 0 e portanto a soma das probabilidades em 254 e 0.
'''


# _______________Funcao histeq (Decima quarta questao)
def histeq(imagem):
    dimensoes = size(imagem)
    histograma = hist(imagem)
    npixels = dimensoes[0] * dimensoes[1]  # Numero de pixels

    try:
        probabilidade = [[0 for x in range(3)] for y in range(256)]

        for x in xrange(0, 256):
            probabilidade[x][0] = histograma[x][0] / (npixels * 1.0)
            probabilidade[x][1] = histograma[x][1] / (npixels * 1.0)
            probabilidade[x][2] = histograma[x][2] / (npixels * 1.0)

        histograma_equalizado = [[0 for x in range(3)] for y in range(256)]
        histograma_equalizado[0][0] = probabilidade[0][0]
        histograma_equalizado[0][1] = probabilidade[0][1]
        histograma_equalizado[0][2] = probabilidade[0][2]

        for x in xrange(1, 256):
            histograma_equalizado[x][0] = probabilidade[x][0] + histograma_equalizado[x - 1][0]
            histograma_equalizado[x][1] = probabilidade[x][1] + histograma_equalizado[x - 1][1]
            histograma_equalizado[x][2] = probabilidade[x][2] + histograma_equalizado[x - 1][2]

        return histograma_equalizado

    except:
        probabilidade = [0.0 for x in range(0, 256)]

        for x in range(0, 256):
            probabilidade[x] = histograma[x] / (npixels * 1.0)

        histograma_equalizado = [0 for x in range(0, 256)]
        histograma_equalizado[0] = probabilidade[0]

        for x in range(1, 256):
            histograma_equalizado[x] = probabilidade[x] + histograma_equalizado[x - 1]

        return histograma_equalizado


# _______________________________________________________Questao 15
def convolve(img, mask):
    tamanho = size(img)  # Tamanho da imagem
    tamanho_mask = size(mask)  # Tamanho da mascara

    newImg = img.copy()
    t_superior = (tamanho_mask[1] - 1) / 2
    s_superior = (tamanho_mask[0] - 1) / 2
    aux2 = 0

    for s in range(0, s_superior * 2):
        for t in range(0, t_superior * 2):
            aux2 += mask[s][t]

    try:
        # Vai armazenar o valor do pixel com os 3 canais em um vetor de 3 posicoes
        aux = [0 for x in range(0, 3)]

        for x in xrange(0, tamanho[1]):  # Altura
            for y in xrange(0, tamanho[0]):  # Comprimento
                for s in range(0, s_superior * 2):
                    for t in range(0, t_superior * 2):
                        # Se extrapolar, substitui pelo valor do pixel da borda
                        if (x + s < 0 or y + t < 0 or x + s >= tamanho[1] or y + t >= tamanho[0]):
                            aux[0] += (newImg[x][y][0] * mask[s][t])  # R
                            aux[1] += (newImg[x][y][1] * mask[s][t])  # G
                            aux[2] += (newImg[x][y][2] * mask[s][t])  # B

                        else:
                            aux[0] += (newImg[x + s][y + t][0] * mask[s][t])  # R
                            aux[1] += (newImg[x + s][y + t][1] * mask[s][t])  # G
                            aux[2] += (newImg[x + s][y + t][2] * mask[s][t])  # B

                newImg[x][y] = aux
                aux = 0
        return newImg
    except:
        aux = 0
        for x in xrange(0, tamanho[1]):  # Altura
            for y in xrange(0, tamanho[0]):  # Comprimento
                for s in range(0, s_superior * 2):
                    for t in range(0, t_superior * 2):
                        # Se extrapolar, substitui pelo valor do pixel da borda
                        if (x + s < 0 or y + t < 0 or x + s >= tamanho[1] or y + t >= tamanho[0]):
                            aux += (img[x][y] * mask[s][t])
                        else:
                            aux += (img[x + s][y + t] * mask[s][t])
                newImg[x][y] = aux / aux2
                aux = 0
        return newImg


# __________________________________________Questao 16
def maskblur():
    return [[0.0625, 0.1250, 0.0625], [0.125, 0.2500, 0.1250], [0.0625, 0.1250, 0.0625]]


# __________________________________________Questao 17
def blur(img):
    return convolve(img, maskblur())


# __________________________________________Questao 18
def seSquare3():
    return [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


# ___________________________________________Questao 19
def seCross3():
    return [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


# -------------------testando funcoes------------------------

scarlet = imread('gostosa.jpg')
scarletG = imread('gostosa.tif')
lenaG = imread('lena.jpg')
lena = imread('lena_std.tif')
# print scarlet #segunda questao letra a
# print scarletG #segunda questao letra b
# print imread ('50x50.gif') #segunda questao letra c
# print nchannels(scarlet) #terceira questao com RGB
# print nchannels(scarletG) #terceira questao com escala de cinza
# print size (scarletG) #imprimindo quarta questao
# print rgb2gray(scarlet)#quinta questao
# print rgb2gray(scarletG) #quinta questao
# print imreadgray(scarlet) #sexta questao
# imshow(scarlet) #setima questao
# print imreadgray(scarletG) #sexta questao
# imshow(scarlet) #setima questao
# imshow(scarletG) #setima questao
# imshow(thresh(scarlet, 100)) #oitava questao
# imshow(negative(scarlet)) #nona questao
# imshow(contrast(scarlet, 3.0, 128))
# showhist(hist(lenaG))
# showhist2(hist(lenaG),5)
# histeq(scarlet)
# showhist2(hist(scarlet),5)
# Filtro de media
filtro1 = [[0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04],
           [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04]]
# Filtro de blur
filtro2 = [[0.0625, 0.1250, 0.0625], [0.125, 0.2500, 0.1250], [0.0625, 0.1250, 0.0625]]
filtro3 = [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]]
filtro4 = [[0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020], [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020],
           [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020], [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020],
           [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020], [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020],
           [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020]]


# print lenaG
# imshow((scarletG))
# print scarletG[1][1]
# imshow(lena)
# imshow( blur(lena))
# imshow(convolve(lena,filtro1))


def hist(imagem):
    try:
        histograma = np.zeros(shape=(3, 256))
        for red in np.nditer(imagem[:, :, 0]):
            histograma[0][red] += 1
        for green in np.nditer(imagem[:, :, 1]):
            histograma[1][green] += 1
        for blue in np.nditer(imagem[:, :, 2]):
            histograma[2][blue] += 1
    except IndexError:
        histograma = np.zeros(shape=(256))
        for gray in np.nditer(imagem):
            histograma[gray] += 1
    return histograma
