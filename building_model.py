import joblib 
x = joblib.load("tfidf_matrix.pkl")
labels = joblib.load("labels.pkl")
print(labels)
x = x[:-1]          
labels = labels[:-1]


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Suppose your labels are stored in a list called 'y'
# Example:
# y = ['happy', 'sad', 'happy', 'angry', ...]
# Make sure this list aligns with the number of sentences in x

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Initialize the model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(x_train, y_train)

# Make predictions
y_pred = nb_model.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

## ssaving the model
joblib.dump(nb_model , "nb_model.pkl")
