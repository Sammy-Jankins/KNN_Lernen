from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np

x_Werte = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
       26., 27., 28., 29., 30.]
y_Werte = [10.01658801, 12.44031689, 16.74158448, 16.99873647, 19.9301948 ,
       21.3662462 , 21.81029917, 28.12350724, 23.99691104, 30.37132938,
       30.7411985 , 27.51173669, 29.86051921, 31.83605475, 30.53806554,
       38.67317334, 48.65172013, 49.65403085, 47.05943994,
       56.90034954, 60.5995608 , 49.83008854, 58.24880042, 61.19101233,
       61.06640885, 57.78297408, 58.12449899, 65.83760744, 63.55813398,
       73.22576714]
x_Werte_fake = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15, 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
       26., 27., 28., 29., 30.]
y_Werte_fake = [10.01658801, 12.44031689, 16.74158448, 16.99873647, 19.9301948 ,
       21.3662462 , 21.81029917, 28.12350724, 23.99691104, 30.37132938,
       30.7411985 , 27.51173669, 29.86051921, 31.83605475, 30.53806554,
       60, 38.67317334, 48.65172013, 49.65403085, 47.05943994,
       56.90034954, 60.5995608 , 49.83008854, 58.24880042, 61.19101233,
       61.06640885, 57.78297408, 58.12449899, 65.83760744, 63.55813398,
       73.22576714]

def pol(x):
    p = 0
    l = []
    for j in range(len(x_Werte_fake)):
        l.append(1)
        for i in range(len(x_Werte_fake)):
            if i == j:
                continue
            l[j] = l[j] * (x-x_Werte_fake[i])/(x_Werte_fake[j]-x_Werte_fake[i])               
    for j in range(len(x_Werte_fake)):
        p = p + l[j]*y_Werte_fake[j]
    return p    
        
plt.scatter(x_Werte, y_Werte)
y_Werte2=[]
for x in x_Werte:
    y_Werte2.append(pol(x))
plt.plot(x_Werte, y_Werte2, color='red')
plt.show()

x_Werte2 = np.linspace(-2,33,500)
y_Werte2=[]
for x in x_Werte2:
    y_Werte2.append(pol(x))
plt.plot(x_Werte2, y_Werte2, color='red')
plt.scatter(x_Werte, y_Werte)
plt.xlim(-1, 31)
plt.ylim(-30, 100)
plt.figure(figsize=(100,100))
plt.show()
