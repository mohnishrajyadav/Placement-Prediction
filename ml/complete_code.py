import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_loader import load_data

# Load dataset
df = load_data()

# Split dataset
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
from sklearn.preprocessing import StandardScaler

# Numerical columns
num_cols = [
    'CGPA',
    'Internships',
    'Projects',
    'Workshops'
]

# Create scaler
scaler = StandardScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(train_df[num_cols])

# Scale test data using the same scaler
X_test_scaled = scaler.transform(test_df[num_cols])

# Print learned statistics
print("Learned means:", scaler.mean_.round(2))
print("Learned std:", scaler.scale_.round(2))
print("=================================================================================")


def onehot():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    # Load Dataset
    df = load_data()
    df = clean_data(df)

    nominal_cols = [
        'Gender',
        'City',
        'Stream',
        'Specialization',
        'HistoryOfBacklogs'
    ]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    ohe = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )

    train_ohe = ohe.fit_transform(train_df[nominal_cols])

    test_ohe = ohe.transform(test_df[nominal_cols])

    feature_names = ohe.get_feature_names_out(nominal_cols)

    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=feature_names,
        index=train_df.index
    )

    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=feature_names,
        index=test_df.index
    )

    print("Encoded Feature Names")
    print(feature_names)

    print("==========================================================")

    print("Training Encoded Data")
    print(train_ohe_df.head())

    print("==========================================================")

    print("Testing Encoded Data")
    print(test_ohe_df.head())

    return train_ohe_df, test_ohe_df

def simple_linear_regression(df):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        x = df[['CGPA']]
        y = df['Internships']
        train_x, test_x, train_y, test_y = train_test_split(
            x, y,
            test_size=0.2,
            random_state=0
        )
        model = LinearRegression()
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        mse = mean_squared_error(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)
        print("MSE:", mse)
        print("R2 Score:", r2)
        print()
df = load_data()
simple_linear_regression(df)

def gradient_descent(X, y, learning_rate=0.001, iterations=100):
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = len(y)
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for i in range(iterations):
        y_pred = theta0 + theta1 * X
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(error * X)
        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1
    return theta0, theta1, mse_history
def decision_tree(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Features
    X = df[
        [
            'CGPA',
            'Internships',
            'Projects',
            'Workshops'
        ]
    ]

    # Target
    y = df['PlacementStatus']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Decision Tree
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        random_state=42
    )
    # Train model
    model.fit(X_train, y_train)
    # Prediction
    y_pred = model.predict(X_test)
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("DECISION TREE CLASSIFIER")
    print("Accuracy:", accuracy)
    #print("\nClassification Report:")
    #print(classification_report(y_test, y_pred))
#LINEAR REGRESSION
X = df['CGPA']
y = df['SalaryPackage']
theta0, theta1, mse_history = gradient_descent(X, y)
print("Theta 0:", theta0)
print("Theta 1:", theta1)
print("Final MSE:", mse_history[-1])
# Decision tree
decision_tree(df)