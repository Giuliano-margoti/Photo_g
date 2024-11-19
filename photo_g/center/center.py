import numpy as np
import matplotlib.pyplot as plt
from photo_g.utils import data_cut
from scipy.optimize import minimize

class Center:
    ''' Define o frame de trabalho para cubo de dados
    parametros:
    ---------
    data_cube: ndarray
        Cubo de dados contendo múltiplos frames (imagens).
    '''
    def __init__(self, data_cube):
        self.data_cube = data_cube
        self.xc_final = None
        self.yc_final = None

    def momento(self, frame_index, xc, yc, r):
        ''' Calcula o momento para um frame específico do cubo '''
        xc = int(xc)
        yc = int(yc)
        data = self.data_cube[frame_index]
        
        # Corta a imagem ao redor do alvo com um raio r
        data_corte = data_cut(data, xc, yc, r)
        
        # Calcula os momentos da intensidade na imagem cortada
        I_i = np.sum(data_corte, axis=0)
        I_j = np.sum(data_corte, axis=1)
        Ii_mean = np.sum(I_i) / len(I_i)
        Ij_mean = np.sum(I_j) / len(I_j)
    
        x_i = np.arange(data_corte.shape[1])
        y_j = np.arange(data_corte.shape[0])
        mask_i = (I_i - Ii_mean) > 0
        mask_j = (I_j - Ij_mean) > 0
    
        xc_new = np.sum((I_i - Ii_mean)[mask_i] * x_i[mask_i]) / np.sum((I_i - Ii_mean)[mask_i])
        yc_new = np.sum((I_j - Ij_mean)[mask_j] * y_j[mask_j]) / np.sum((I_j - Ij_mean)[mask_j])
    
        # Ajusta o centro de corte com base no corte feito em data_corte
        self.xc_final = xc - r + xc_new
        self.yc_final = yc - r + yc_new
        
        return self.xc_final, self.yc_final

    def plot_center(self, frame_index, r=10):
        ''' Plota o centro calculado em um frame específico do cubo '''
        data = self.data_cube[frame_index]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(data, cmap='gray', origin='lower')
        axs[0].plot(self.xc_final, self.yc_final, '+', color='red', markersize=10)

        # Plot da imagem cortada com zoom de 10 pixels
        data_corte = data_cut(data, int(self.xc_final), int(self.yc_final), r)
        axs[1].imshow(data_corte, cmap='gray', origin='lower')
        axs[1].plot(-int(self.xc_final)+self.xc_final+r, -int(self.yc_final)+self.yc_final+r, '+', color='red', markersize=10)  # Centro marcado no corte
        plt.tight_layout()
        plt.show()

    @staticmethod
    def gaussian_with_background(xy, x0, y0, sigma_x, sigma_y, I0, bg, circular=False):
        x, y = xy
        if circular:
            # Usar sigma único para Gaussiana circular
            gaussian = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_x**2))
        else:
            # Usar sigma_x e sigma_y para Gaussiana elíptica
            gaussian = I0 * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
        return gaussian + bg
    
    def fit_gaussian_2d(self, frame_index, x_guess, y_guess, circular=False):
        ''' Ajuste de Gaussiana em 2D para um frame específico do cubo '''
        data = self.data_cube[frame_index]
        
        if circular:
            # Parâmetros iniciais para Gaussiana circular (sigma único)
            initial_guess = (x_guess, y_guess, 2.5, np.max(data), np.median(data))
        else:
            # Parâmetros iniciais para Gaussiana elíptica (sigma_x e sigma_y diferentes)
            initial_guess = (x_guess, y_guess, 2.5, 2.5, np.max(data), np.median(data))
    
        # Coordenadas interpoladas
        xy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        xy = (xy[0].ravel(), xy[1].ravel())
    
        # Função de erro para ajustar
        def residuals(params):
            if circular:
                # Chamar a Gaussiana com a opção circular
                model = self.gaussian_with_background(xy, params[0], params[1], params[2], params[2], params[3], params[4], circular=True)
            else:
                # Chamar a Gaussiana sem a opção circular
                model = self.gaussian_with_background(xy, *params)
            return np.ravel(model - data.ravel())
    
        # Ajuste usando minimize com o método Powell
        result = minimize(lambda params: np.sum(residuals(params)**2), initial_guess, method='Powell')
    
        optimized_params = result.x
        self.xc_final, self.yc_final = optimized_params[0], optimized_params[1]
        
        if circular:
            sigma = optimized_params[2]
            amplitude = optimized_params[3]
            background = optimized_params[4]
            return self.xc_final, self.yc_final, sigma, amplitude, background
        else:
            sigma_x = optimized_params[2]
            sigma_y = optimized_params[3]
            amplitude = optimized_params[4]
            background = optimized_params[5]
            return self.xc_final, self.yc_final, sigma_x, sigma_y, amplitude, background
