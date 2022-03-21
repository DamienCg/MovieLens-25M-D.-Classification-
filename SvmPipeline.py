from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
import Utility as utility


def SVCCLassifier(X_train, X_test, y_train, y_test,class_names):
    model = SVC(decision_function_shape='ovo', probability=True)

    PipelineIMB = Pipeline([
        ('sample', 'passthrough'),
        ('svm', model)
    ])

    """ Define search space """
    param_grid = {
        'sample': [
            SMOTE(k_neighbors=3),
            RandomOverSampler(),
        ],
        'svm__C': np.arange(1, 20, 1),
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    grid_imba = GridSearchCV(PipelineIMB, param_grid, cv=kf, scoring='f1_weighted',
                             verbose=10, n_jobs=-1, error_score='raise')

    grid_imba.fit(X_train, y_train)
    y_pred = grid_imba.predict(X_test)

    utility.printScore(y_test, y_pred, grid_imba.best_params_)
    utility.Print_Conf_Matrix(grid_imba, X_test, y_test, y_pred, class_names)


