import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

file_path = r'.\sms-spam-collection-dataset\spam.csv'
df = pd.read_csv(file_path,encoding='latin-1')
print(df.head())
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"class", "v2":"text"})
print(df.head())
df["class"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
x = df['text']
y = df['label']
print(df.head())
cv = CountVectorizer()
x = cv.fit_transform(x) # Fit the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
def predict():
    message = input("Enter a message to classify\n")
    data=[message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    if my_prediction==1:
        print("SPAM")
    else:
        print("NOT SPAM / HAM")
while True:
    print("1 - Classify\n2- Quit\n")
    choice = input()
    if choice=="1":
        predict()
    elif choice=="2":
        break
    else:
        print("Invalid choice, please choose again\n")

print("Thank you")
