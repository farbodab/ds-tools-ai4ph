import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    url = "https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv"
    all_cases = pd.read_csv(url)
    
    agg = all_cases.groupby(['Accurate_Episode_Date'])['Row_ID'].count().to_frame().reset_index()
    agg['Accurate_Episode_Date'] = pd.to_datetime(agg['Accurate_Episode_Date'])
    agg['t-1'] = agg['Row_ID'].shift(1)
    agg['t-2'] = agg['Row_ID'].shift(2)
    agg['t-3'] = agg['Row_ID'].shift(3)
    agg['t-4'] = agg['Row_ID'].shift(4)
    agg['t-5'] = agg['Row_ID'].shift(5)
    agg['t-6'] = agg['Row_ID'].shift(6)
    agg['t-7'] = agg['Row_ID'].shift(7)
    agg['t'] = agg['Row_ID']
    agg.dropna(how='any',inplace=True)
    
    x = agg[['t-1','t-2','t-3','t-4','t-5','t-6','t-7']]
    y = agg['t']

    test_size = 0.3
    dataset_size = len(agg)
    test_index = int(test_size * dataset_size)


    x_train, x_test, y_train, y_test = x[:test_index], x[test_index:], y[:test_index], y[test_index:]
    
    regr = LinearRegression()
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test)
    print(y_pred)
