import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import tensorflow as tf
from flask import Flask
from flask_script import Manager

# Create Flask app
app = Flask(__name__)
# Create Manager class
manager = Manager(app)

@manager.command
def CNN_predict():
    print('Tensorflow-gpu version = ', tf.__version__)

    # input from keyboard
    input_X = []
    print('Hi, I am the Oracle, I can estimate the price per square of London house')
    print('What is the year when the sale of house to be completed?')
    input_X.append(int(input()))
    print('What is the type of the proper? 0: Terraced, 1: Semi-Detached, 2: Flats, 3: Detached')
    input_X.append(int(input()))
    print('What is the tenure of property? 0: Leasehold, 1: Freehold')
    input_X.append(int(input()))
    print('What is the total floor area?')
    input_X.append(int(input()))
    print('What is the number of habitable rooms?')
    input_X.append(int(input()))
    # get system time
    t_sys = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Part1 Load and clean dataset

    filePath = './London_house_price.csv'  # data path

    london = pd.read_csv(filePath)  # load dataset
    # print(london.info) # info of dataset
    describe = london.describe()

    # print(describe)

    def scatter_plot(x, y, xlabel, ylabel):
        plt.scatter(x, y)
        plt.title('%s - House Prices' % xlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.autoscale(enable=True)
        plt.savefig('./%s.png' % xlabel)
        plt.clf()

    scatter_plot(london['year'], london['priceper'], 'year', 'House Prices')
    scatter_plot(london['propertytype'], london['priceper'], 'propertytype', 'House Prices')
    scatter_plot(london['duration'], london['priceper'], 'duration', 'House Prices')
    scatter_plot(london['classt'], london['priceper'], 'classt', 'House Prices')
    scatter_plot(london['CURRENT_ENERGY_EFFICIENCY'], london['priceper'], 'CURRENT_ENERGY_EFFICIENCY', 'House Prices')
    scatter_plot(london['POTENTIAL_ENERGY_EFFICIENCY'], london['priceper'], 'POTENTIAL_ENERGY_EFFICIENCY',
                 'House Prices')
    scatter_plot(london['numberrooms'], london['priceper'], 'numberrooms', 'House Prices')

    # Calculate the corr
    corr = london.corr()
    # print(corr)
    # Plot the corr
    varcorr = london[london.columns].corr()
    mask = np.array(varcorr)
    mask[np.tril_indices_from(mask)] = False
    sn.heatmap(varcorr, mask=mask, vmax=1, vmin=-1, square=True, annot=False, cmap='rainbow')
    plt.savefig('./corr.png')
    plt.clf()

    # Part2 Build up models

    X = pd.DataFrame(
        np.c_[london['year'], london['propertytype'], london['duration'], london['tfarea'], london['numberrooms']])
    Y = london['priceper']
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train_copy = X_train

    # ----------------------------LinearRegression model----------------------------
    model_LR = LinearRegression(normalize=True)
    model_LR.fit(X_train,Y_train)

    # intercept of model
    print(model_LR.intercept_)
    # coeffcients of model
    coeffcients_LR = pd.DataFrame([X_train.columns,model_LR.coef_]).T
    coeffcients_LR = coeffcients_LR.rename(columns={0:'Attribute',1:'Coeffcients'})
    print(coeffcients_LR)

    # ------------------------------Lasso--------------------------------------------
    model_Lasso = LassoCV()
    model_Lasso.fit(X_train,Y_train)
    alpha = model_Lasso.alpha_
    print('Best alpha from LassoCV: ' + str(alpha))

    model_Lasso = Lasso(alpha)
    model_Lasso.fit(X_train,Y_train)

    # --------------------------------ANN--------------------------------------------
    def normalize_CNN(x):
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean) / std
        return x

    def plot_mae(log, t_sys):
        plt.plot(log['mae'])
        plt.plot(log['val_mae'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('./model_acc_' + t_sys + '.jpg')
        plt.show()
        plt.clf()

    def plot_loss(log, t_sys):
        plt.plot(log['loss'])
        plt.plot(log['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('./model_loss_' + t_sys + '.jpg')
        plt.show()
        plt.clf()

    X_train = normalize_CNN(X_train)  # Normalized X_train
    X_train = np.expand_dims(X_train, axis=2)  # Expand the dimention of X_train
    X_test = normalize_CNN(X_test)  # Normalized X_test
    X_test = np.expand_dims(X_test, axis=2)  # Expand the dimention of X_test
    n_input = X_train.shape[1]
    # print(n_input)
    # CNN model build
    model_cnn = tf.keras.models.Sequential()
    # step 1 - Convolution
    model_cnn.add(tf.keras.layers.Conv1D(128, 3, input_shape=(n_input, 1), activation='relu'))
    model_cnn.add(tf.keras.layers.Conv1D(512, 3, activation='relu'))
    # step 3 - Dropout
    model_cnn.add(tf.keras.layers.Dropout(0.3))
    # step 4 - Flatten
    model_cnn.add(tf.keras.layers.Flatten())
    # step 5 - output
    model_cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model_cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model_cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model_cnn.add(tf.keras.layers.Dense(units=1))

    # Compiling the CNN
    model_cnn.compile(optimizer='adam', loss='mse', metrics='mae')
    # Training the CNN on the Training set and evaluating it on the Test set
    model_cnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1)

    input_X = (input_X - np.mean(X_train_copy))/np.std(X_train_copy)  # Normalized X_train
    input_X = np.expand_dims(input_X, axis=0)  # Expand the dimention of X_train
    input_X = np.expand_dims(input_X, axis=-1)  # Expand the dimention of X_train
    output_Y = str(model_cnn.predict(input_X))
    output_Y = output_Y.replace('[','')
    output_Y = output_Y.replace(']','')
    print('Estimate house price per square meter:', output_Y, 'Pounds')

    # Time series forcasting
    # average for year from 1985 to 2021
    dict_predict = {}
    dict_ts = {2016: 6494.604936024998, 2002: 2836.189941158246, 1996: 1168.607277348888, 2020: 6681.066648716291,
               2012: 4790.668186854992, 2015: 6068.883390675356, 2006: 3769.8531282168997, 2001: 2451.29383280776,
               1995: 1105.563581076889, 2003: 3103.8194510282297, 1999: 1843.251044220226, 2013: 5104.0236629588435,
               2011: 4615.949811511135, 2010: 4511.98761161587, 2007: 4264.407704898497, 2005: 3462.015681775603,
               2017: 6582.746462577941, 2004: 3393.5665875054124, 2018: 6515.243257689478, 2009: 4065.2939900876227,
               2014: 5744.496354382676, 2000: 2222.9966446855396, 2008: 4307.686629345853, 2019: 6391.240003473369,
               1998: 1573.110719714926, 1997: 1364.2290414983242, 2021: 7008.399584413637}
    input_peopertytype = 0
    input_duration = 0
    input_tfarea = 120
    input_numberrooms = 4

    for year in range(2022,2050):
        input_X = [year,input_peopertytype,input_duration,input_tfarea,input_numberrooms]
        input_X = (input_X - np.mean(X_train_copy))/np.std(X_train_copy)  # Normalized X_train
        input_X = np.expand_dims(input_X, axis=0)  # Expand the dimention of X_train
        input_X = np.expand_dims(input_X, axis=-1)  # Expand the dimention of X_train
        output_Y = model_cnn.predict(input_X)
        dict_predict[year] = float(output_Y)

    # Plot time series forcasting
    x1_ts = range(1995,2022)
    x2_ts = range(2022,2050)
    y1_ts = []
    y2_ts = []
    for i in x1_ts:
        y1_ts.append(dict_ts[i])
    for i in x2_ts:
        y2_ts.append(dict_predict[i])
    plt.plot(x1_ts,y1_ts,'o-',color='r',label='Train')
    plt.plot(x2_ts,y2_ts,'o-',color='y',label='Prediction')
    plt.xlabel('Year')
    plt.ylabel('Price per square (Pounds)')
    plt.legend(loc = 'best')
    plt.savefig('./ts.jpg')
    plt.clf()

    # plot
    #data_log = model_cnn.history.history
    #plot_mae(data_log, t_sys)
    #plot_loss(data_log, t_sys)

    # Part3 Evaluate models
    # R-squared evaluate
    # print('R-Squared: %.4f'% model_Lasso.score(X_test,Y_test))
    # Plot the evaluation
    # price_pred = model_LR.predict(X_test)
    # price_pred = model_Lasso.predict(X_test)
    # plt.scatter(Y_test,price_pred)
    # plt.grid()
    # plt.xlabel('Actual Prices')
    # plt.ylabel('Predicted Prices')
    # plt.title('Actual Prices vs Predicted Prices')
    # plt.savefig('./R_Squared.png')
    # plt.clf()

if __name__ == '__main__':
    manager.run()
    #app.run(debug=True)