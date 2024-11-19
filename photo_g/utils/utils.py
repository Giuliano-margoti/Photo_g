import numpy as np

def information(data, method=np.mean):
    
    sky_val = {}

    sky_val['method used'] = method(data)
    sky_val['max'] = np.max(data)
    sky_val['min'] = np.min(data)
    
    return sky_val

def data_cut(data, xc, yc, r):
    
    box = [int(yc-r), int(yc+r), int(xc-r), int(xc+r)]
    
    data_box = data[box[0]:box[1], box[2]:box[3]]
    
    return data_box