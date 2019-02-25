import numpy as np

def fourier(imagem):
    largura = imagem.shape[0]
    altura = imagem.shape[1]
    imagem_saida = np.zeros_like(imagem)

    for u in range(0, largura):
        for v in range(0, altura):
            soma = 0
            for x in range(0, largura):
                for y in range(0, altura):
                    soma += imagem[x, y] * (np.cos(2 * np.pi * ((u * x) / largura + (v * y) / altura) - 1j * np.sin(
                        2 * np.pi * ((u * x) / largura + (v * y) / altura))))
                    print(soma)
            imagem_saida[u, v] = soma / (altura * largura)

    return imagem_saida


def fourier_inversa(imagem):
    largura = imagem.shape[0]
    altura = imagem.shape[1]
    imagem_saida = np.zeros_like(imagem)

    for x in range(0, largura):
        for y in range(0, altura):
            soma = 0
            for u in range(0, largura):
                for v in range(0, altura):
                    soma += imagem[u, v] * (np.cos(2 * np.pi * ((u * x) / largura + (v * y) / altura) + 1j * np.sin(
                        2 * np.pi * ((u * x) / largura + (v * y) / altura))))

            imagem_saida[x, y] = soma
    return imagem_saida
