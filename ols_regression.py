import numpy as np
import matplotlib.pyplot as plt
import random as random


class OLS():
    def __init__(self, x, y):
        '''
        This function performs a univariate ordinary least squares regression, mapping a dependent variable to independent through two model parameters. Returned is a function that can be evaluated at any "x" value.
        '''
        # Create input array, compute covariance matrix and variance in x
        self.n = np.size(x)
        array_data = np.zeros((2, self.n))
        array_data[0,:] = x; array_data[1,:] = y 

        self.xbar = np.mean(x)
        self.ybar = np.mean(y)
        self.array_data = array_data
        self.covs = np.cov(array_data)
        self.Var_x = np.var(x)

        # Slope of OLS fit given by covariance of x and y divided by the variance of x = (E[XY] - E[X]E[Y]) / (E[X^2] - E[X]^2)
        self.model_parameters = np.zeros(2)
        self.model_parameters[1] = self.covs[1][0] / self.covs[0,0] 
        self.model_parameters[0] = self.ybar - self.model_parameters[1] * self.xbar

        # Lambda function is returned that can evaluate dependent variable given an input variable
        estimator = lambda x: self.model_parameters[0] + self.model_parameters[1] * x
        self.estimator = estimator

        self.__estimator_analysis()
        return
    
    def __estimator_analysis(self):
        '''
        Evaluate the relevant statistics of the OLS estimator. This includes the following:
            self.Var_model_parameters: The residual variance of the model parameters (slope and intercept).
            self.TSS: Total sum of squares
            self.RSS: Residual sum of squares
            self.ESS: Error sum of squares
            self.R_squared: Coefficient of determination
            self.r: Correlation coefficient
        Stores them in OLS class.
        '''
        self.TSS = np.sum((self.array_data[1,:] - np.mean(self.array_data[1,:]))**2)
        self.RSS = np.sum((self.estimator(self.array_data[0,:]) - np.mean(self.array_data[1,:]))**2)
        self.ESS = np.sum((self.estimator(self.array_data[0,:]) - self.array_data[1,:])**2)
        self.R_squared = self.RSS / self.TSS
        self.r = self.covs[1,0] / (np.sqrt(self.covs[0,0]) * np.sqrt(self.covs[1,1]))
        self.Var_model_parameters = np.zeros(2)
        var_residual = self.ESS / (self.n - 2.)
        self.Var_model_parameters[1] = var_residual / (self.n * self.Var_x)
        self.Var_model_parameters[0] = var_residual * (np.sum(self.array_data[0,:]**2) / (self.n **2 * self.Var_x))
        return

    def evaluate(self, x):
        '''
        Uses the OLS estimator to evaluate the dependent variable for a prescribed independent variable
        '''
        return self.estimator(x)

    def statistics(self):
        '''
        Prints the statistics of the Ordinary Least Square (OLS) estimator
        '''
        print('''
        OLS Statistics
        ------------------------------------------------------------
        Total sum of squares (TSS): {:.2f}
        Residual sum of squares (RSS): {:.2f}
        Error sum of squares (ESS): {:.2f}
        Coefficient of determination (0 to 1), R^2: {:.2f}
        Correlation coefficient (-1 to 1), r: {:.2f}
        Standard error in slope model parameter, b1: {:.2f}
        Standard error in intercept model parameter, b0: {:.2f}
        ------------------------------------------------------------
        '''.format(self.TSS, self.RSS, self.ESS, self.R_squared, self.r, self.Var_model_parameters[1], self.Var_model_parameters[0]))
        return


if __name__=='__main__':
    # Test the OLS algorithm by solving the Corn-Fertiliser problem
    y = np.array([6., 10., 12., 14., 16., 18., 22., 24., 26., 32.])
    x = np.array([40., 44., 46., 48., 52., 58., 60., 68., 74., 80.])
    r = OLS(x,y)
    
    # Print out relevant statistics
    r.statistics()

    # Plotting results, currently commented out
    plt.scatter(x,y)
    ax = plt.gca()
    ax.set_xlabel('Fertiliser Price ($)')
    ax.set_ylabel('Corn Price ($)')
    plt.plot(x, r.evaluate(x))
    plt.show()
    

