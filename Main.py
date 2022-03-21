import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from NbPipeline import NBCLassifier
from RandomForestPipeline import RFCLassifier
from SvmPipeline import SVCCLassifier

"""Import Dataset"""
dataset = pd.read_csv('DatasetMovies.csv', sep=',')

"""Split Data and target"""
X = dataset.iloc[:, 1:-1]
y = dataset['rating']
class_names = np.unique(y)

"""Label Encoder"""
y  = LabelEncoder().fit_transform(y)

""" LDA """
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X = lda.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, stratify=y)

NBCLassifier(X_train, X_test, y_train, y_test,class_names)
#RFCLassifier(X_train, X_test, y_train, y_test,class_names)
#SVCCLassifier(X_train, X_test, y_train, y_test,class_names)

