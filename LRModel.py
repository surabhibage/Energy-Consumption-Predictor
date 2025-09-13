import numpy as np
import pandas as pd
import matplotlib.pyplot as py

class LinearRegression:
    # def __init__(self, df):
    #     self.df = df

    def createArrays(df):

        data_list = ['SQFTEST', 'KWH', 'MONEYPY']
        ydata_name = 'KWH'
        column_names = df.columns.tolist()

        column_names_remove = list(filter(lambda item: item not in data_list, column_names))
        df = df.drop(columns = column_names_remove)

        condition = (df['SQFTEST'] <= 5000) & (df['KWH'] <= 50000)
        no_outliers = df[condition]
        y = np.array(no_outliers[ydata_name].to_list())

        no_outliers = no_outliers.drop(ydata_name, axis=1)

        # Maximum value per column
        max_values = no_outliers.max()
        # Minimum value per column
        min_values = no_outliers.min()

        column_arrays = [no_outliers[col].tolist() for col in no_outliers.columns]
        x = column_arrays

        return y, x, max_values, min_values


    def formatArrays(y_array, array_of_xs):
    
        y_matrix = y_array.reshape(-1,1)

        noOfRows = y_array.size
        x_col1 = np.full((noOfRows, 1), 1) 

        x_matrix = np.matrix(array_of_xs).T

        x_matrix = np.hstack((x_col1, x_matrix))
        
        return y_matrix, x_matrix


    def lineOfBestFit(y_matrix, x_matrix): #an array of numpy arrays 

        xT_matrix = x_matrix.transpose()
        xT_x = np.dot(xT_matrix, x_matrix)

        xT_y = np.dot(xT_matrix, y_matrix)

        xT_x_inverse = np.linalg.inv(xT_x)

        c_matrix = np.dot(xT_x_inverse, xT_y)

        y_matrix_prediction = np.dot(x_matrix, c_matrix)

        return y_matrix_prediction, c_matrix


    def plotLine(x , predictions, y_data, c_matrix, max_values, min_values):

        fig = py.figure()
        ax = fig.add_subplot(111, projection='3d')
        x1 = np.linspace(min_values['MONEYPY'], max_values['MONEYPY'], 1000)
        x2 = np.linspace(min_values['SQFTEST'], max_values['SQFTEST'], 1000)
        x1, x2 = np.meshgrid(x1, x2)
        y = c_matrix[0,0] + c_matrix[1,0]*x1 + c_matrix[2,0]*x2
        ax.plot_surface(x1, x2, y, alpha=0.5)
        ax.set_xlabel('MONEYPY')
        ax.set_ylabel('SQFTEST')
        ax.set_zlabel('KWH')

        # Defining x values for a horizontal line
        x = np.linspace(0, 16, 100)
        y = np.full_like(x, 3000) 
        z = np.full_like(x, 15000)  
        ax.plot(x, y, z)
        py.show()

        fig = py.figure()
        py.plot(predictions, predictions)
        py.scatter(y_data, np.asarray(predictions).reshape(-1,1), marker = '.', color = 'red')
        py.show()
        

    def linearRegression(df):
        y, x, max_values, min_values = createArrays(df)
        y_matrix, x_matrix = formatArrays(y, x)
        y_matrix_prediction, c_matrix = lineOfBestFit(y_matrix, x_matrix)
        plotLine(x, y_matrix_prediction, y, c_matrix, max_values, min_values)



