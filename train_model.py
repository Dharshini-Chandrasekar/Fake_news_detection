import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
fake_data = pd.read_csv('fake.csv')
true_data = pd.read_csv('true.csv')

# Add labels to the datasets
fake_data['label'] = 'FAKE'
true_data['label'] = 'TRUE'

# Combine the datasets
data = pd.concat([fake_data, true_data])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = data['text'] 
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')

