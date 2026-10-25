import unittest
import numpy as np
from src.analysis.final_model_validation import design_at, loss_by_hour


class FinalEvaluationTests(unittest.TestCase):
    def test_hourly_loss_is_mean_of_individual_errors(self):
        # A mean prediction curve would incorrectly produce zero error here.
        y=np.zeros((2,4));p=np.array([[1]*4,[-1]*4])
        rows=loss_by_hour(y,p,np.array([700,700]),'GRU',42,'holdout')
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['mae_loss']==1 and r['mse_loss']==1 and r['n']==2 for r in rows))

    def test_experiment_design_uses_identical_twenty_lags(self):
        x=np.arange(2*200*4).reshape(2,200,4)
        design=design_at(x,np.array([1]),np.array([170]))
        self.assertEqual(design.shape,(1,20))
        np.testing.assert_array_equal(design[0,:5],x[1,[169,168,167,145,1],0])
