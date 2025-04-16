from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# === Step 1: Load the dataset ===
dataset = load_dataset("batterydata/pos_tagging")
train_data = dataset["train"].select(range(1000))
test_data = dataset["test"]

# === Step 2: Load GloVe embeddings ===
def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings("glove.6B.50d.txt")

# === Step 3: Word to embedding function ===
def word_to_embedding(word, embeddings, dim=50):
    return embeddings.get(word.lower(), np.zeros(dim))

# === Step 4: Prepare features and labels ===
def prepare_data(data, embeddings_dict, dim=50):
    X, y = [], []
    for tokens, tags in zip(data['words'], data['labels']):
        for word, tag in zip(tokens, tags):
            X.append(word_to_embedding(word, embeddings_dict, dim))
            y.append(tag)
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_data, glove_embeddings)
X_test, y_test = prepare_data(test_data, glove_embeddings)

# === Step 5: Encode labels ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
# Add UNK handling for test labels
known_labels = set(label_encoder.classes_)
y_test_enc = []

unk_index = len(label_encoder.classes_)  # One extra index for unknowns

for label in y_test:
    if label in known_labels:
        y_test_enc.append(label_encoder.transform([label])[0])
    else:
        y_test_enc.append(unk_index)  # Assign to UNK class

y_test_enc = np.array(y_test_enc)


# === Step 6: Train Classifiers ===

# Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train_enc)
y_pred_log = log_reg.predict(X_test)

# SVM Classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train_enc)
y_pred_svm = svm_model.predict(X_test)

# === Step 7: Evaluate ===
log_acc = accuracy_score(y_test_enc, y_pred_log)
svm_acc = accuracy_score(y_test_enc, y_pred_svm)

print(f"Logistic Regression Accuracy: {log_acc * 100:.2f}%")
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
