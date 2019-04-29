import numpy as np
import matplotlib.pyplot as plt

#Ein Input-Neuron
#Anzahl_verborgene_Neuronen = 4
#Bias = "Ja" # mögliche Werte: "Ja", "Nein"
#Aktivierungsfunktion = "ReLU" # mögliche Werte: "ReLU", "Linear"
#Input_Gewichte = [1,1,2,8]#[-1,2]#[1,1,2,8]
#Bias_Gewichte = [0,1,-2,-16]#[0,-2]#[0,1,-2,-16]
#Output_Gewichte = [1,1,1,1]

#Daten = [Anzahl_verborgene_Neuronen, Bias, Aktivierungsfunktion, Input_Gewichte, Bias_Gewichte, Output_Gewichte]

def Act(x, Aktivierungsfunktion):
    if Aktivierungsfunktion == "ReLU":
        return x * (x > 0)
    elif Aktivierungsfunktion == "Linear":
        return x
    elif Aktivierungsfunktion == "Sigmoid":
        return 1/(1+np.exp(-x))
    else:
        raise ValueError("Aktivierungsfunktion falsch definiert.")
    
def KNN_Funktion(x, Daten):
    [Anzahl_verborgene_Neuronen, Bias, Aktivierungsfunktion, Input_Gewichte, Bias_Gewichte, Output_Gewichte] = Daten
    if Anzahl_verborgene_Neuronen != len(Input_Gewichte):
        raise ValueError("Anzahl der verborgenen Neuronen und Input-Gewichte stimmen nicht überein.")
    if Anzahl_verborgene_Neuronen != len(Output_Gewichte):
        raise ValueError("Anzahl der verborgenen Neuronen und Output-Gewichte stimmen nicht überein.")
    if (Anzahl_verborgene_Neuronen != len(Bias_Gewichte)) & (Bias == "Ja"):
        raise ValueError("Anzahl der verborgenen Neuronen und Bias-Gewichte stimmen nicht überein.")
    Zustand = []
    for j in range(Anzahl_verborgene_Neuronen):
        Zustand.append(Output_Gewichte[j] * Act(Input_Gewichte[j] * x + Bias_Gewichte[j], Aktivierungsfunktion))
    return np.sum(Zustand)

def Graph_KNN_Funktion(Daten, links, rechts):
    x_Werte = np.linspace(-1,2,100)
    y_Werte = []
    for j in range(len(x_Werte)):
        y_Werte.append(KNN_Funktion(x_Werte[j], Daten))
    
    plt.plot(x_Werte, y_Werte, color='blue', label='KNN-Funktion')
    plt.legend()
    plt.title('KNN-Funktion')
    plt.show()  

def Graph_KNN_Funktion_Expo(Daten):
    x_Werte = np.linspace(-3,3,100)
    y_Expo = np.exp(x_Werte)
    y_Werte = []
    for j in range(len(x_Werte)):
        y_Werte.append(KNN_Funktion(x_Werte[j], Daten))
    
    plt.plot(x_Werte, y_Werte, color='blue', label='KNN-Funktion')
    plt.plot(x_Werte, y_Expo, color='red', label='Exponentialfunktion')
    plt.legend()
    plt.title('KNN-Funktion und Exponentialfunktion')
    plt.show()  
    maxi = 0

    for j in range(len(x_Werte)):
        maxi = max(maxi, np.abs(y_Werte[j]-y_Expo[j]))
    print("Der Approximationsfehler beträgt: " + str("{:.4f}".format(maxi)))
    if maxi < 2:
        print("Der Fehler ist kleiner als 2, Glückwunsch!")
    else:
        print("Leider ist der Fehler größer als 2 :(")

#Graph_KNN_Funktion_Expo(Daten)


    
