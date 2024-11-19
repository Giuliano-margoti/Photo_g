import numpy as np
import matplotlib.pyplot as plt
import warnings
from photo_g.utils import information
from photo_g.utils import Read

class Photometry:
    ''' Lugar para informações da fotometria
    parametros:
    ---------
    cube : ndarray
        Cubo de dados 3D com as imagens.
    '''
    
    def __init__(self, data_cube):
        self.data_cube = data_cube
            
        self.information = information(data_cube)
        self.mask_valid_pixels_sky = None
        self.mask_valid_pixels_tar = None
       	self.x, self.y = Read.xy(np.ones(data_cube[0].shape))
       	 
    def sky_ring(self, frame_index, xc, yc, inner_radius_ring, radius_ring, star_threshold):
        ''' Calcula o valor médio do céu em um anel, excluindo pixels acima de um limiar (estrelas). '''
        
        data = self.data_cube[frame_index]
        mask_ring = ((np.sqrt((self.x - xc)**2 + (self.y - yc)**2) >= inner_radius_ring) & 
                     (np.sqrt((self.x - xc)**2 + (self.y - yc)**2) <= (inner_radius_ring+radius_ring)))
        
        mask_valid_pixels = mask_ring & (data <= star_threshold)
        self.mask_valid_pixels_sky = mask_valid_pixels
        
        sky_mean = data[mask_valid_pixels].mean() if np.any(mask_valid_pixels) else np.nan
        
        if np.isnan(sky_mean):
            sky_mean = 0
            warnings.warn("\nThe sky mean (sky_mean) is NaN, modified to zero!\n"
                          "This occurs because:\n"
                          "inner_radius is greater than outer_radius or the region has too many stars, "
                          "resulting in no acceptable sky region. Use a lower threshold.")
            
        return sky_mean

    def eliptical_aperture(self, frame_index, xc, yc, inner_radius_ring, radius_ring, a, b, angle, saturate_cut=np.inf):
        data = self.data_cube[frame_index]
        sky_mean = self.sky_ring(frame_index, xc, yc, inner_radius_ring, radius_ring, star_threshold=self.information['method used'] * 1.5)
        
        theta = np.radians(angle)
        
        x_rot = (self.x - xc) * np.cos(theta) + (self.y - yc) * np.sin(theta)
        y_rot = -(self.x - xc) * np.sin(theta) + (self.y - yc) * np.cos(theta)
        
        mask = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1 & (data <= saturate_cut)
        self.mask_valid_pixels_tar = mask
        
        
        
        if np.any(mask & self.mask_valid_pixels_sky):
            warnings.warn("The aperture contains pixels in the sky background ring. Adjust the parameters to avoid overlap.")
        
        fluxos = np.sum(data[mask])
        npixs = len(data[mask])
        
        fluxos_new = fluxos - npixs * sky_mean
        
        snr = fluxos_new / np.sqrt(fluxos_new + npixs * sky_mean) if fluxos_new + npixs * sky_mean > 0 else 0
        
        self.xx = xc
        self.yy = yc
        
        return fluxos_new, sky_mean, snr
    
    def rectangular_aperture(self, frame_index, xc, yc, inner_radius_ring, radius_ring, width, height, angle):
        
        data = self.data_cube[frame_index]
        sky_mean = self.sky_ring(frame_index, xc, yc, inner_radius_ring, radius_ring, star_threshold=self.information['method used'] * 1.5)
        
        theta = np.radians(angle)
        
        x_shifted = self.x - xc
        y_shifted = self.y - yc
        x_rot = x_shifted * np.cos(theta) + y_shifted * np.sin(theta)
        y_rot = -x_shifted * np.sin(theta) + y_shifted * np.cos(theta)
        
        x_min = -width / 2
        x_max = width / 2
        y_min = -height / 2
        y_max = height / 2
        
        mask = (x_rot >= x_min) & (x_rot <= x_max) & (y_rot >= y_min) & (y_rot <= y_max)
        self.mask_valid_pixels_tar = mask
        
        if np.any(mask & self.mask_valid_pixels_sky):
            warnings.warn("The aperture contains pixels in the sky background ring. Adjust the parameters to avoid overlap.")
        
        fluxos = np.sum(data[mask])
        npixs = len(data[mask])
        
        fluxos_new = fluxos - npixs * sky_mean
        snr = fluxos_new / np.sqrt(fluxos_new + npixs * sky_mean) if fluxos_new + npixs * sky_mean > 0 else 0
        
        self.xx = xc
        self.yy = yc
        
        return fluxos_new, sky_mean, snr

    def plot_images(self, frame_index=0):
        # Selecionando o frame
        data = self.data_cube[frame_index]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        yy = int(round(self.yy))
        xx = int(round(self.xx))

        # Garantindo que os índices estejam dentro dos limites da imagem
        y1, y2 = max(0, yy - 10), min(data.shape[0], yy + 10)
        x1, x2 = max(0, xx - 10), min(data.shape[1], xx + 10)
        
        # Exibindo a imagem cortada no primeiro subplot
        axes[0].imshow(data[y1:y2, x1:x2], cmap='gray', origin='lower')
        axes[1].imshow(data[y1:y2, x1:x2], cmap='gray', origin='lower')
        
        # Criando highlighted_data com NaNs
        highlighted_data = np.full_like(data, np.nan)
        
        # Aplicando máscaras
        if self.mask_valid_pixels_sky is not None:
            highlighted_data[self.mask_valid_pixels_sky] = self.information['max']
        
        if self.mask_valid_pixels_tar is not None:
            highlighted_data[self.mask_valid_pixels_tar] = self.information['max']
        
        # Exibindo dados destacados no segundo subplot
        axes[1].imshow(highlighted_data[y1:y2, x1:x2], cmap='jet', origin='lower', alpha=0.5)  
        
        # Exibindo os valores dos índices para verificação
        print(f"self.xx = {self.xx}, self.yy = {self.yy}")
        print(f"Slicing: y = {y1}:{y2}, x = {x1}:{x2}")
    
        plt.show()


        
    def abertura(self, frame_index, pixels, inner_radius_ring, radius_ring):
        
        data = self.data_cube[frame_index]
        xc = np.mean([p[0] for p in pixels])
        yc = np.mean([p[1] for p in pixels])
        
        sky_mean = self.sky_ring(frame_index, xc, yc, inner_radius_ring, radius_ring, star_threshold=self.information['method used'] * 1.5)
        
        mask = np.zeros_like(data, dtype=bool)
        for x, y in pixels:
            mask[int(y), int(x)] = True
        self.mask_valid_pixels_tar = mask
        
        if np.any(mask & self.mask_valid_pixels_sky):
            warnings.warn("The aperture contains pixels in the sky background ring. Adjust the parameters to avoid overlap.")
        
        fluxos = np.sum(data[mask])
        npixs = len(data[mask])
        
        fluxos_new = fluxos - npixs * sky_mean
        snr = fluxos_new / np.sqrt(fluxos_new + npixs * sky_mean) if fluxos_new + npixs * sky_mean > 0 else 0
        
        self.xx = xc
        self.yy = yc
        return fluxos_new, sky_mean, snr
