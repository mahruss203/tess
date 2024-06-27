import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def train_and_save_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
    column_names = ["Sequence Name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Class"]
    df = pd.read_csv(url, delim_whitespace=True, names=column_names)

    feature_columns = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
    X = df[feature_columns].values
    y = df["Class"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    joblib.dump((classifier, scaler, label_encoder), 'model.pkl')
    print('Model and scaler saved to model.pkl')

if __name__ == "__main__":
    train_and_save_model()
