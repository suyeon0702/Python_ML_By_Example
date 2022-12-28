### Chapter 2 : 나이브 베이즈를 이용한 영화 추천 엔진 구축


import numpy as np
from collections import defaultdict


#2.3.1 밑바닥부터 구현하는 나이브 베이즈

# 데이터셋 정의
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])

#
def get_label_indices(labels):
    """
    레이블과 반환 인덱스 기준으로 샘플 그룹화
    @param labels: 레이블 리스트
    @return label_indices: dictionary, {class1: [indices], class2: [indices]}
    """

    label_indices = defaultdict(list)
    for index , label in enumerate(labels):
        label_indices[label].append(index)

    return label_indices

label_indices = get_label_indices(Y_train)
print('label_indices:\n', label_indices)


def get_prior(label_indices):
    """
    훈련 샘플을 이용한 사전확률 계산
    @param label_indices: 클래스별로 그룹화된 샘플 인덱스
    @retrun prior: dictionary, key = 클래스 레이블, value = 사전확률
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

prior = get_prior(label_indices)
print('Prior:', prior)


def get_likelihood(features, label_indices, smoothing=0):
    """
    훈련 샘플을 이용한 우도 계산
    @param features: 특징 행렬
    @param label_indices: 클래스별로 그룹화된 샘플 인덱스
    @param smoothing: int, 가산(additive) 평활화 계수
    @return likelihood: dictionary, key = 클래스,
                        value = 조건부 확률 P(feature|class) 벡터
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print('Likelihood:\n', likelihood)


def get_posterior(X, prior, likelihood):
    """
    사전확률과 우도를 이용한 테스트 샘플의 사후확률 계산
    @param X: 테스트 샘플
    @param prior: dictionary, key = 클래스 레이블, value = 사전확률
    @param likelihood: dictionary, key = 클래스 레이블, value = 조건부 확률 벡터
    @return posteriors: dictionary, key = 클래스 레이블, value = 사후확률
    """
    posteriors = []
    for x in X:
        # 사후확률은 사전확률 * 우도에 비례한다.
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1-likelihood_label[index])

        # 모든 합이 1이 되도록 정규화한다.
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

posterior = get_posterior(X_test, prior, likelihood)
print('Posterior:\n', posterior)


#2.3.2 사이킷런을 이용한 나이브 베이즈 구현
from sklearn.naive_bayes import BernoulliNB


clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities:\n', pred_prob)


pred = clf.predict(X_test)
print('[scikit-learn] Prediction:', pred)


#### 2.4 나이브 베이즈를 이용한 영화 추천기 구축
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data_path = r"C:\Users\mr2022029\tmp\RPA_nb\Python Machine Learning By Example\ml-1m\ratings.dat"
n_users = 6040
n_movies = 3706


def load_rating_data(data_path, n_users, n_movies):
    """
    파일에서 평점 데이터를 로드하고 각 영화의 평점과
    이에 대응되는 인덱스(movie_id) 반환
    @param data_path: 평점 데이터 파일의 경로
    @param n_users: 관객 수
    @param n_movies: 평점을 받은 영화의 수
    @return data: 넘파이 배열([user, movie]), 평점 데이터
    @return movie_n_rating: dictionary, {movie_id: 평점 수};
    @return movie_id_mapping: dictionary, {movie_id: 평점 데이터의 열(column) 인덱스}
    """
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping

data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')

display_distribution(data)


movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')


X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)



display_distribution(Y)


recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(Y_train), len(Y_test))


clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])


prediction = clf.predict(X_test)
print(prediction[:10])


accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy*100:.1f}%")


#### 2.5 분류 성능 평가
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


print(confusion_matrix(Y_test, prediction, labels=[0, 1]))


print(precision_score(Y_test, prediction, pos_label=1))
print(recall_score(Y_test, prediction, pos_label=1))
print(f1_score(Y_test, prediction, pos_label=1))


f1_score(Y_test, prediction, pos_label=0)


report = classification_report(Y_test, prediction)
print(report)


pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            # truth와 prediction이 모두 1인 경우
            if y == 1:
                true_pos[i] += 1
            # truth는 0이고 prediction은 1인 경우
            else:
                false_pos[i] += 1
        else:
            break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()
true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]


import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


roc_auc_score(Y_test, pos_prob)


from sklearn.model_selection import StratifiedKFold


k = 5
k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train, Y_train)
            prediction_prob = clf.predict_proba(X_test)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)

for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f'    {smoothing}     {fit_prior}     {auc/k:.5f}')
        
        
clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))
