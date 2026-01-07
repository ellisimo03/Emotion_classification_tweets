import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/emotion.csv")  # opens data set, stores in df

def cleaning(text): #this function will clean the data
    text = str.lower(text) #converts text to lower
    text = re.sub(r"\s+", " ", text).strip() #removes white space
    return text

df["cleanText"] = df["text"].apply(cleaning) #applies cleaning func

#70% train
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"], # STRATIFY: equal amount of labels from dataset
    random_state=42 # ensure consistent output everytime (not random)
)

#split tempdf into 15% val and 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

#cleaned input the model learns from and the labels for supervised learning
X_train = train_df["cleanText"]
X_val   = val_df["cleanText"]
X_test  = test_df["cleanText"]

y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # uses unigrams and bigrams
    max_features=10000   # limits size
)

# looks at training text only, converts to numerical vector
X_train_vec = vectorizer.fit_transform(X_train)

# uses idf weights from train data, converts test text into vector
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# model training
model = LogisticRegression(
    max_iter=500,
    class_weight="balanced"
)

#trains classifer using training data
model.fit(X_train_vec, y_train)

#predicts validation set for model selection
val_preds = model.predict(X_val_vec)
#predicts unseen data on the test set
test_preds = model.predict(X_test_vec)

#outputs results and confusion matrix
print("Validation results:")
print(classification_report(y_val, val_preds, digits=3))
print("Test results:")
print(classification_report(y_test, test_preds, digits=3))
print(confusion_matrix(y_test, test_preds))
