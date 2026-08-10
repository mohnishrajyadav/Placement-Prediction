import pandas as pd
from sklearn.model_selection import train_test_split


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