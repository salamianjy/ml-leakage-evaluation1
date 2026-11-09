"""Demonstrate target leakage using a post-outcome feature."""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=17

def build_data(n=1200):
    X,y=make_classification(n_samples=n, n_features=8, n_informative=4,
                            n_redundant=2, flip_y=.08, class_sep=.8,
                            random_state=SEED)
    rng=np.random.default_rng(SEED)
    # A post-outcome label recorded after the decision: unavailable at prediction time.
    post_outcome=np.where(rng.random(n)<.995,y,1-y)
    X_leaky=np.column_stack([X,post_outcome])
    return X,y,X_leaky

def evaluate(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,
                                  stratify=y,random_state=SEED)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=SEED))
    pipe.fit(X_train,y_train)
    scores=pipe.predict_proba(X_test)[:,1]
    pred=(scores>=.5).astype(int)
    return {'roc_auc':round(float(roc_auc_score(y_test,scores)),4),
            'accuracy':round(float(accuracy_score(y_test,pred)),4),
            'precision':round(float(precision_score(y_test,pred,zero_division=0)),4),
            'recall':round(float(recall_score(y_test,pred,zero_division=0)),4),
            'test_rows':len(y_test)}

def compare():
    X,y,leaky=build_data()
    return {'valid_features':evaluate(X,y),'with_leakage':evaluate(leaky,y)}

def main():
    result=compare()
    print(json.dumps(result,indent=2))
    out=Path(__file__).parent/'outputs';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print('Warning: the leaked feature is recorded AFTER the outcome and cannot be used at prediction time.')

if __name__=='__main__':main()
