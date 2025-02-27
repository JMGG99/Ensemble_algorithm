import pandas as pd
from sklearn.ensemble import VotingClassifier # para clasificación
from sklearn.ensemble import VotingRegressor # para regresión
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Importamos modelos, incluido el meta modelo VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import VotingClassifier

# Cargamos los datos
cancer = pd.read_csv('https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/cancer.csv')

# obtenemos dummies
cancer_dummies = (pd.get_dummies(cancer,
                                columns = ['diagnosis'],
                                drop_first = [True],
                                prefix = ['D'])
                  .loc[:,['D_M','radius_mean', 'concave points_mean']]
                  )

# obtenemos los arreglos numpy
X = cancer_dummies.copy().drop(columns=['D_M']).values
y = cancer_dummies.copy()['D_M'].values

# partimos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.30,
                                                    random_state= 123)

# Instanciamos los clasificadores individuales
lr = LogisticRegression(random_state=123)
knn = KNN()
dt = DecisionTreeClassifier(random_state=123)

# Definimos una lista de tuplas con (nombre_del_clasificador, clasificador)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]


for clf_name, clf in classifiers:
    # Ajustamos sobre train
    clf.fit(X_train, y_train)
    # Generamos pronósticos con test
    y_pred = clf.predict(X_test)
    # Evaluamos las métricas
    print('-'*30)
    print(clf_name)
    print('accuracy: ', accuracy_score(y_test, y_pred))
    print('precision:', precision_score(y_test, y_pred))
    print('recall:   ', recall_score(y_test, y_pred))
    print('f1:       ', f1_score(y_test, y_pred))

# Instanciamos el VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Ajustamos sobre train
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
# Evaluamos las metricas
print('-'*30)
print('ensambe VotingClassifier')
print('accuracy: ', accuracy_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('recall:   ', recall_score(y_test, y_pred))
print('f1:       ', f1_score(y_test, y_pred))