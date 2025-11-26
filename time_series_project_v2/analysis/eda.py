
def perform_eda(df):
    print("\n--- EDA Summary ---")
    print(df.describe())
    print(df.head())
    print("Missing values:", df.isnull().sum())
