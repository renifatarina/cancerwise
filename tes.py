import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('survey_lung_cancer.csv')

df['GENDER'].replace(['M', 'F'], [0, 1], inplace=True)
df['LUNG_CANCER'].replace(['NO', 'YES'], [0, 1], inplace=True)

df = df.drop(['SMOKING', 'SHORTNESS OF BREATH'], axis=1)
df.drop_duplicates(inplace=True)


Q1 = df['AGE'].quantile(0.25)
Q3 = df['AGE'].quantile(0.75)
IQR  = Q3 - Q1
Lower_bound = Q1-1.5*IQR
Upper_bound = Q3+1.5*IQR
New_Age= (df['AGE']>Lower_bound) & (df['AGE']<Upper_bound)
filtered_data= df[New_Age]
y = filtered_data['LUNG_CANCER']
X = filtered_data.drop('LUNG_CANCER',axis=1)

for i in X.columns[2:]:
    temp=[]
    for j in X[i]:
        temp.append(j-1)
    X[i]=temp

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 42)

scaler = StandardScaler()
X_train['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_val['AGE'] = scaler.transform(X_val[['AGE']])
X_test['AGE'] = scaler.transform(X_test[['AGE']])

# Membuat model ANN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(15, activation='relu', input_shape=[13]),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Melatih model
model.fit(X_train, y_train,
                    epochs=100, batch_size=16,
                    validation_data=(X_val, y_val),
                    shuffle=True)


model.save('model.h5')