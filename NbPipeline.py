from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
import Utility as utility


def NBCLassifier(X_train, X_test, y_train, y_test,class_names):

    """This is a imblearn pipeline """
    PipelineIMB = Pipeline([
        ('sample', 'passthrough'),
        ('NB', GaussianNB())
    ])

    # define search space
    param_grid = {
        'sample': [
            SMOTE(k_neighbors=3),
            RandomOverSampler(),
        ],
        'NB__var_smoothing': np.logspace(0, -9, num=1000)
    }

    """Cross validation"""
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    grid_imba = GridSearchCV(PipelineIMB, param_grid, cv=kf, scoring='f1_weighted', verbose=10, n_jobs=-1,
                             error_score='raise')

    grid_imba.fit(X_train, y_train)

    y_pred = grid_imba.predict(X_test)

    utility.printScore(y_test, y_pred, grid_imba.best_params_)
    utility.Print_Conf_Matrix(grid_imba, X_test, y_test, y_pred, class_names)




