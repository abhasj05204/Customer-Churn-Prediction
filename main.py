import pandas as pd

def main():
    df=pd.read_csv("data/telco_churn.csv")

    print(df.head())
    print(df.shape)

if __name__== "__main__":
    main()
print("Datset loaded successfully")