import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# Function for text preprocessing
stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Load dataset
@st.cache_data
def load_data():
    news_dataset = pd.read_csv('C:/Users/HP/Desktop/Fake News Detection/train.csv')
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    return news_dataset

# Prepare data for training
data = load_data()
X = data['content'].values
Y = data['label'].values

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=200)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Y, test_size=0.2, stratify=Y, random_state=2)

# Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM Model
history = model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_test, Y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)

# Accuracy scores during training
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Streamlit UI
st.title("Fake News Detection")
st.sidebar.title("Input")

# Text input for index
index_str = st.sidebar.text_input("Enter index number of news article:", type="default")

# Convert the input to integer
try:
    index = int(index_str)
    if index < 0 or index >= len(Y_test):
        st.error("Index is out of range")
    else:
        # Perform prediction using the index
        X_new = X_test[index].reshape(1, -1)
        prediction = model.predict(X_new)[0][0]
        if prediction == 0:
            st.success("The news is Real")
        else:
            st.error("The news is Fake")
except ValueError:
    st.error("Please enter a valid integer index")

# Display accuracy score
st.write(f"Accuracy Score: {accuracy:.2f}")

# Display accuracy scores in text format
st.subheader("Accuracy Scores")
st.write(f"Final Test Accuracy: {accuracy:.4f}")

st.write("Accuracy Scores During Training:")
for epoch, acc in enumerate(train_accuracy, start=1):
    st.write(f"Epoch {epoch}: {acc:.4f}")

# Plotting the accuracy curve
st.subheader("Training and Validation Accuracy")
epochs = range(1, len(train_accuracy) + 1)
fig, ax = plt.subplots()
ax.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
ax.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
ax.set_title('Training and Validation Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)
