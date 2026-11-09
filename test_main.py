import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import build_data, compare

class TestLeakage(unittest.TestCase):
    def test_extra_feature_matches_outcome_mostly(self):
        X,y,leaky=build_data()
        self.assertEqual(leaky.shape[1],X.shape[1]+1)
        self.assertGreater(np.mean(leaky[:,-1]==y),.98)
    def test_leakage_inflates_auc(self):
        result=compare()
        self.assertGreater(result['with_leakage']['roc_auc'],result['valid_features']['roc_auc']+.05)
        self.assertEqual(result['valid_features']['test_rows'],300)

if __name__=='__main__':unittest.main()
