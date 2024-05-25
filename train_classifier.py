import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
# 80% of the data will be used for training and 20% for testing
# The split is stratified to maintain the class distribution in both sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Use the trained model to predict labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
