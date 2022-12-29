### Chapter 3 : 서포트 벡터 머신을 이용한 얼굴 인식

#3.1.4 SVM 구현

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
print('Input data size :', X.shape)
print('Output data size:', Y.shape)
print('Label names:', cancer_data.target_names)
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)



from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, random_state=42)

clf.fit(X_train, Y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=42,
    shrinking=True, tol=0.001, verbose=False)

accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy*100:.1f}%")



#3.1.5 시나리오 4: 두 개 이상의 클래스 처리

from sklearn.datasets import load_wine
from sklearn.metrics import classification_report



from sklearn.datasets import load_wine

wine_data = load_wine()
X = wine_data.data
Y = wine_data.target
print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('Label names:', wine_data.target_names)
n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()
print(f'{n_class0} class0 samples, \n{n_class1} class1 samples, \n{n_class2} class2 samples.')



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=42,
    shrinking=True, tol=0.001, verbose=False)

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')



from sklearn.metrics import classification_report

pred = clf.predict(X_test)
print(classification_report(Y_test, pred))



#3.1.6 시나리오 5: 커널을 통한 선형으로 분리할 수 없는 문제 해결

import numpy as np

import matplotlib.pyplot as plt
X = np.c_[# 양성 클래스
          (.3, -.8),
          (-1.5, -1),
          (-1.3, -.8),
          (-1.1, -1.3),
          (-1.2, -.3),
          (-1.3, -.5),
          (-.6, -1.1),
          (-1.4, 2.2),
          (1, 1),
          # 음성 클래스
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [-1] * 8 + [1] * 8



import matplotlib.pyplot as plt

gamma_option = [1, 2, 4]
for i, gamma in enumerate(gamma_option, 1):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X, Y)
    plt.scatter(X[:, 0], X[:, 1], c=['b']*8+['r']*8, zorder=10, cmap=plt.cm.Paired)
    plt.axis('tight')
    XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
    Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.title('gamma = %d' % gamma)
    plt.show()
    
    
    
    #### 3.2 SVM을 이용한 얼굴 이미지 분류
    
    #3.2.1 얼굴 이미지 데이터 셋
