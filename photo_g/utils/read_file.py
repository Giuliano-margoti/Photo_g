import numpy as np
import astropy.io.fits as fits

class Read:
    ''' 
    Classe para leitura de arquivos FITS para fotometria.
    '''
    
    @staticmethod
    def read_fits_file(img_name):
        ''' Lê o arquivo FITS e retorna os dados como um array numpy '''
        data = fits.getdata(img_name).astype(float)
        return data  # Retorna diretamente o array 3D para um cubo ou 2D para uma imagem única
    
    @staticmethod
    def xy(data):
        ''' 
        Gera e retorna as coordenadas x e y para o array fornecido.
        Para um cubo de dados (3D), retorna as coordenadas para o primeiro frame.
        '''
        if data.ndim == 3:  # Verifica se é um cubo de dados
            y = np.arange(data.shape[1])
            x = np.arange(data.shape[2])
        else:
            y = np.arange(data.shape[0])
            x = np.arange(data.shape[1])
        
        x, y = np.meshgrid(x, y)
        return x, y
