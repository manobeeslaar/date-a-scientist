import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import metrics

#functions
#condencing the data in religion
def condence(data, column):
    data_string = data[column].values
    for index in range(len(data_string)):
        if data_string[index].find('agnosticism') != -1:
            data_string[index] = 'agnosticism'
        if data_string[index].find('atheism') != -1:
            data_string[index] = 'atheism'
        if data_string[index].find('buddhism') != -1:
            data_string[index] = 'buddhism'
        if data_string[index].find('catholicism') != -1:
            data_string[index] = 'catholicism'
        if data_string[index].find('christianity') != -1:
            data_string[index] = 'christianity'
        if data_string[index].find('hinduism') != -1:
            data_string[index] = 'hinduism'
        if data_string[index].find('islam') != -1:
            data_string[index] = 'islam'
        if data_string[index].find('judaism') != -1:
            data_string[index] = 'judaism'
        if data_string[index].find('other') != -1:
            data_string[index] = 'other'
        
#Create your df here:
df = pd.read_csv('profiles.csv')
df = df.replace(np.nan, 'none', regex=True)

condence(df, 'religion')

orientation = df.orientation.unique()
orientation_qty = df.orientation.value_counts().values

#plotting orientation
fig, ax = plt.subplots()
ax.barh(orientation, 
        orientation_qty, 
        align = 'center',
        color = '#0080FF',
        ecolor = 'black')

ax.set_yticklabels(orientation)
ax.invert_yaxis()
ax.set_xlabel('Qty ot People')
ax.set_title('Sexual Orientation')
plt.savefig('orientation.png')
plt.show()

religion = df.religion.unique()
religion_qty = df.religion.value_counts().values

#plotting religion
fig, ax = plt.subplots(figsize = (20, 15))
ax.barh(religion,
        religion_qty,
        align = 'center',
        color = '#0080FF',
        ecolor = 'black')

ax.set_yticklabels(religion, fontsize = 7.8)
ax.invert_yaxis()
ax.set_xlabel('Qty of Poeple')
ax.set_title('Religion/ Preferred Faith')
plt.savefig('religion.png')
plt.show()

diet = df.diet.unique()

religion_mapping = {
        'agnosticism': 0,
        'atheism': 1,
        'buddhism': 2,
        'catholicism': 3,
        'christianity': 4,
        'hinduism': 5,
        'islam': 6,
        'judaism': 7,
        'other': 8,
        'none': 9,
        }

df['religion_code'] = df.religion.map(religion_mapping)

diet_mapping = {
        'strictly anything': 0,
        'mostly other': 1,
        'anything': 2,
        'vegetarian': 3,
        'mostly anything': 4,
        'mostly vegetarian': 5,
        'strictly vegan': 6,
        'strictly vegetarian': 7,
        'mostly vegan': 8,
        'strictly other': 9,
        'mostly halal': 10,
        'other': 11,
        'vegan': 12,
        'mostly kosher': 13,
        'strictly halal': 14,
        'halal': 15,
        'strictly kosher': 16,
        'kosher': 17,
        'none': 18}

df['diet_code'] = df.diet.map(diet_mapping)

orientation_mapping = {
        'straight': 0,
        'bisexual': 1,
        'gay': 2}

df['orientation_code'] = df.orientation.map(orientation_mapping)

feature_data = df[['diet_code', 'religion_code', 'orientation_code']]

x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data_scaled = pd.DataFrame(x_scaled, columns = feature_data.columns)

#diet given religion

X = df['religion']
Y = df['diet']
colors = np.random.rand(59946)
plt.scatter(X, Y, c=colors, alpha=0.5)
plt.xticks(rotation = 90)
plt.savefig('Diet and Religion.png')
plt.show()


print('Regression mlr diet given religion')
Xx = feature_data[['religion_code']]
Yy = feature_data[['diet_code']]

x_train, x_test, y_train, y_test = train_test_split(Xx, Yy)

mlr = LinearRegression()

start = time.time()
mlr.fit(x_train, y_train)
end = time.time()
print('mlr fit time: ', end - start)

print('mlr score', mlr.score(x_test, y_test))

y_predicted_mlr = mlr.predict(x_test)

#print('mlr metrics: ', metrics.classification_report(y_test, y_predicted_mlr.astype(int)))

#KN Regressor diet given religion
print('KN Regressor diet given religion')

regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')

start = time.time()
regressor.fit(x_train, y_train)
end = time.time()
print('knr regressor fit time: ', end - start)

print('knr score', regressor.score(x_test, y_test))

y_predicted_knr = regressor.predict(x_test)

#print('knr metrics: ', metrics.classification_report(y_test, y_predicted_knr.astype(int)))


#guess religion given orientation
datapoints = df[['diet_code', 'religion_code']]

labels = df[['orientation_code']]

x_train, x_test, y_train, y_test = train_test_split(datapoints, labels)

print('KNeighborClassifier')

knn = KNeighborsClassifier(n_neighbors=5)

start = time.time()
knn.fit(x_train, y_train)
end = time.time()
print('knn fit time: ', end - start)

y_predicted = knn.predict(x_test)
score = knn.score(x_test, y_test)
print('knn score: ', score)

print('SVC Classifier')

classifier = SVC(kernel='rbf', gamma=0.1)

start = time.time()
classifier.fit(x_train, y_train)
end = time.time()
print('svc fit time: ', end - start)

y_predicted_svc = classifier.predict(x_test)
score_svc = classifier.score(x_test, y_test)
print('SVC score: ', score_svc)
#print('SVC metrics: ', metrics.classification_report(y_test, y_predicted_svc))



 




