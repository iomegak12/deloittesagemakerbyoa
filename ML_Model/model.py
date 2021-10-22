import pandas as pd
import numpy as np
import warnings
import pickle

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

df = pd.read_csv('Diabetestype.csv')

df.drop('Class', 1, inplace=True)

x = df.drop('Type', 1)
y = df.iloc[:, -1]

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier(
    n_estimators=50,
    max_features='sqrt')

models = [logreg, logreg_cv, rf]


def compute_score(classifier, x, y, scoring='accuracy'):
    xval = cross_val_score(classifier, x, y, cv=5, scoring=scoring)
    return np.mean(xval)


for model in models:
    print('Cross Validation of {0}'.format(model.__class__))
    score = compute_score(classifier=model, x=x, y=y, scoring='accuracy')
    print('Cross Validation Score = {0}'.format(score))

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=0)

model = rf.fit(x_train, y_train)

model_name = 'model.pkl'

pickle.dump(model, open(model_name, 'wb'))

print('Finished Saving Model!')
