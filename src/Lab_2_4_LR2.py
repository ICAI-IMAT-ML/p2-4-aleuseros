import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        #We make sure both X and y are arrays so we can work with them
        X = np.array(X, dtype=float)
    
        # We make sure X is not 1D
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        
        #Now we apply the usual operations using matrixes
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Invert the matrixes, multipl them
        w = np.linalg.inv(XtX) @ Xty
        
        #Obtain parameters
        self.intercept = w[0]
        self.coefficients = w[1:]


    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            #Calculamos las predicciones del modelo
            predictions = self.intercept +  np.dot(X[:,1:], self.coefficients)

            #Calculamos el error, diferencia entre predicciones y valor real
            error = predictions - y

            #Calculamos los parámetros del gradiente
            gradient_intercept = (2/m)*np.sum(error)
            gradient_coefficients = (2/m)*np.dot(X[:,1:].T, error)

            #Actualizamos los parámetros
            self.intercept -= learning_rate*gradient_intercept
            self.coefficients -= learning_rate*gradient_coefficients

            #Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.mean(error**2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        #Case one, X only has one variable
        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            #Usual linear regresion y = a+bx
            predictions = self.intercept + self.coefficients[0]*X

       #Case two: multivariable prediction     
        else:
            # TODO: Predict when X is more than one variable
            #We use np.dot so that we can include the multiple models (matricial multiplication)
            predictions = self.intercept + np.dot(X, self.coefficients)

        #Finally, we return the prediction   
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO
    r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # Root Mean Squared Error
    # TODO
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    # TODO
    mae = np.mean(np.abs(y_true - y_pred))


    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    #Copiamos el dataset a otro, para no modificar el original
    X_transformed = X.copy()

    #Iteramos los indices categoricos
    for index in sorted(categorical_indices, reverse=True):
        categorical_column = X_transformed[:,index]
        #Obtenemos los valores unicos
        unicos = list(set(categorical_column))
        #vamos poniendo ceros, para cambiarlos luego
        one_hot = [[0]*len(unicos) for _ in range(len(categorical_column))]

        #Cambiamos los 0 por 1 donde sea necesario iterando el dataset
        for i in range(len(categorical_column)):
            for j in range(len(unicos)):
                if unicos[j] == categorical_column[i]:
                    one_hot[i][j] = 1
        
        #Convertimos en un array
        one_hot = np.array(one_hot)

        #si nos piden librarnos de la primera columna lo hacemos
        if drop_first:
            one_hot = one_hot[:, 1:]

        #Separamos en parte izqda y parte dcha y lo concatenamos despues para obtener el dataset final
        izquierda = X_transformed[:, :index]
        derecha = X_transformed[:, index+1:]
        
        X_transformed = np.concatenate([izquierda, one_hot, derecha], axis=1)

    return X_transformed

