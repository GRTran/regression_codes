import unittest
from ols_regression import OLS
import numpy as np

class test_OLS(unittest.TestCase):

    def setUp(self):
        '''
        Set up reference problem for which answers to statstical properties are known
        '''
        x = np.array([6., 10., 12., 14., 16., 18., 22., 24., 26., 32.])
        y = np.array([40., 44., 46., 48., 52., 58., 60., 68., 74., 80.])        
        self.estimator = OLS(x,y)
        return

    def test_R_squared(self):
        '''
        Test the coefficient of determination
        '''
        self.assertAlmostEqual(0.9710, self.estimator.R_squared, delta=5e-5)
        return
    
    def test_r(self):
        '''
        Test the correlation coefficient
        '''
        self.assertAlmostEqual(self.estimator.r, 0.9854, delta=5e-5)
        return

    def test_intercept_variance(self):
        '''
        Test the variance of the intercept model parameter
        '''
        self.assertAlmostEqual(self.estimator.Var_model_parameters[0], 3.92, delta=5e-3)
        return

    def test_slope_variance(self):
        '''
        Test the variance of the slope model parameter
        '''
        self.assertAlmostEqual(self.estimator.Var_model_parameters[1], 0.01, delta=5e-3)
        return

if __name__ == '__main__':
    unittest.main()
