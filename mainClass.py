from LRModel import LinearRegression as lr
import pandas as pd

def main():
    df = pd.read_csv('/Users/surabhibage/Documents/EnergyConsumption2020_csv_copy.csv', header=0)
    lr.linearRegression(df)

if __name__ == "__main__":
    main()