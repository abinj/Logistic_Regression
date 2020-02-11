import pandas as pd

# Import the dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("voice.csv")
print(df.head())

encode = LabelEncoder()
df.label = encode.fit_transform(df.label)

X = df.iloc[:, df.columns != 'label'].values
Y = df.iloc[:, 20].values

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
