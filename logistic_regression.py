import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Import the dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv("voice.csv")
print(df.head())

# Encode Male and Female to 1 and 0
encode = LabelEncoder()
df.label = encode.fit_transform(df.label)

X = df.iloc[:, df.columns != 'label'].values
Y = df.iloc[:, -1].values

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

logreg = LogisticRegression(random_state=0)
# Fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('\nAccuracy score on test data : ', accuracy_score(y_test, y_pred))


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix', cm)
# 0 means Negative and 1 means Positive
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()

plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted label')
plt.show()


#Correlation matrix
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
