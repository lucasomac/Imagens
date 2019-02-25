import fft.fourier as ft
import lab1.lucas_macedo as lc


lc.imshow(lc.imread("excla.gif"))

imagem = lc.imread("excla.gif")

im = ft.fourier(imagem)

lc.imshow(im)

im2 = ft.fourier_inversa(im)

lc.imshow(im2)