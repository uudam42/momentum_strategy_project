import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def load_and_prepare_data(folder_path):
    """
    Go through each file in folder_path that ends with '_features.csv',
    build up feature matrix X and label vector y.
    """
    x_data, y_data = [], []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith("_features.csv"):
            # skip anything that's not a features file
            continue

        # read the CSV into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file_name))

        # make a simple
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # drop any rows with missing data (especially from the shift)
        df.dropna(inplace=True)

        # pull out our three features: RSI, EMA20, MACD
        features = df[['RSI', 'EMA20', 'MACD']].values
        labels   = df['target'].values

        # add to our growing lists
        x_data.extend(features)
        y_data.extend(labels)

    # return as pandas objects for easy integration with sklearn
    x_df = pd.DataFrame(x_data, columns=['RSI', 'EMA20', 'MACD'])
    y_sr = pd.Series(y_data, name='target')
    return x_df, y_sr

def train_model(x_df, y_sr):
    """
    Split data, train a Random Forest, and print out how well it did.
    """
    # quick sanity check
    print(f"Hey, I loaded {len(x_df)} samples with {x_df.shape[1]} features each.")
    print("Here are the first few rows:")
    print(x_df.head())
    print("And label distribution (1=up, 0=down):")
    print(y_sr.value_counts(normalize=True))

    # split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_sr, test_size=0.3, random_state=42
    )

    # train a RandomForest straight-up
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)


    preds  = clf.predict(x_test)
    acc    = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2%}")

    return clf

if __name__ == "__main__":
    # build the path to our features_data folder reliably
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURES_DIR = os.path.join(BASE_DIR, "features_data")

    X, y = load_and_prepare_data(FEATURES_DIR)
    model = train_model(X, y)

    joblib.dump(model, "model/random_forest_model.pkl")
    print(" Model saved to model/random_forest_model.pkl")
