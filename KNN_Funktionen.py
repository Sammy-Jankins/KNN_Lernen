import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
        
def KNN2_Netz1(Kosten):
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x') # Input-Variable x
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y') # Output-Variable y
    a = tf.Variable(tf.random.normal(shape = [1,1]), name='a')
    b = tf.Variable(np.random.normal(), name='b')
    with tf.variable_scope('Output') as scope:
        y_output = tf.add(tf.matmul(x, a), b)
        if Kosten == "Variante 1":
            loss = tf.reduce_sum(tf.square(y_output - y))
        elif Kosten == "Variante 2": 
            loss = tf.reduce_sum(y_output - y)
        else:
            square_residuals = tf.square(y_output - y)
            loss = tf.reduce_sum([square_residuals[0], square_residuals[14]])

    return x, y, y_output, loss

def KNN2_Netz2(Anzahl_verborgene_Neuronen, Aktivierungsfunktion):    
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    b1 = tf.Variable(tf.random.normal(shape = [Anzahl_verborgene_Neuronen]), name='b1')
    
   # bses = []
    #for j in range(Anzahl_verborgene_Neuronen):
       # bses.append(tf.Variable(tf.random.normal(shape = [1]), name='b'+str(j)))
    #b2 = tf.Variable(tf.random.normal(shape = [1]), name='b2')
    #ases = []
    #yses = []
    #for j in range(Anzahl_verborgene_Neuronen):
        #with tf.variable_scope('Verborgene_Schicht_Nr' + str(j)) as scope:
            #ases.append(tf.Variable(tf.random.normal(shape = [1,1]), name='a' + str(j)))
            #yses.append(tf.nn.relu(x * ases[j] + bses[j]))
    
    with tf.variable_scope('Verborgene_Schicht') as scope:
        a1 = tf.Variable(tf.random.normal(shape = [1,Anzahl_verborgene_Neuronen]), name='a1')
        if Aktivierungsfunktion == "ReLU":
            y1 = tf.nn.relu(tf.matmul(x, a1) + b1)
        elif Aktivierungsfunktion == "Sigmoid":
            y1 = tf.nn.sigmoid(tf.matmul(x, a1) + b1)
        else:
            y1 = tf.matmul(x, a1) + b1
        
        
    #ases2 = []
    #for j in range(Anzahl_verborgene_Neuronen):
        #ases2.append(tf.Variable(tf.random.normal(shape = [1,1]), name='a2_'+str(j)))

    
    with tf.variable_scope('Output') as scope:
        a2 = tf.Variable(tf.random.normal(shape = [Anzahl_verborgene_Neuronen,1]), name='a2') 
        y_pred = tf.matmul(y1, a2) 
    #y_pred = yses[0]*ases2[0]
    #for j in range(1,Anzahl_verborgene_Neuronen):
        #y_pred = y_pred + yses[j]*ases2[j]
    loss = tf.reduce_sum(tf.square(y_pred - y))
        #square_residuals = tf.square(y_pred - y)
        #loss = square_residuals[0]+square_residuals[14]


    return x, y, y_pred, loss
