from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline
import Utility as utility


def RFCLassifier(X_train, X_test, y_train, y_test,class_names):

    PipelineIMB = Pipeline([
        ('sample', 'passthrough'),
        ('rf', RandomForestClassifier(random_state=1))
    ])

    # define search space
    param_grid = {
        'sample': [
            SMOTE(k_neighbors=3),
            RandomOverSampler(),
        ],
        'rf__max_features': [2, 3, 4, 5, 6, 7],
        'rf__n_estimators': [200, 300, 400, 500, 600, 700, 800],
        'rf__criterion': ["gini", "entropy"]
    }

    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    grid_imba = GridSearchCV(PipelineIMB, param_grid, cv=kf, scoring='f1_weighted',
                             verbose=10, n_jobs=-1, error_score='raise')

    grid_imba.fit(X_train, y_train)
    y_pred = grid_imba.predict(X_test)

    utility.printScore(y_test, y_pred, grid_imba.best_params_)
    utility.Print_Conf_Matrix(grid_imba, X_test, y_test, y_pred, class_names)



