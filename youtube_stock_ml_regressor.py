import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def to_timestamp(date):
    return date.timestamp()


def fix_timestamps(videos):
    videos['stock_date_timestamp'] = videos['stock_date_before'].apply(to_timestamp)
    videos['upload_date_timestamp'] = videos['Date'].apply(to_timestamp)
    videos['stock_date_after_timestamp'] = videos['stock_date_after'].apply(to_timestamp)

    return videos


def get_videos(folder):
    all_files = glob(folder + "/*.csv")
    
    videos = pd.DataFrame()
    for fileName in all_files:
        df = pd.read_csv(fileName, parse_dates=['Date', 'stock_date_before', 'stock_date_after'])
        videos = pd.concat([videos, df])

    # Get unique videos
    videos = videos.drop_duplicates(subset='videoId', keep='first').dropna()

    return fix_timestamps(videos)


if __name__ == "__main__":
    folder = sys.argv[1]

    videos = get_videos(folder)
    # print(videos.dtypes)

    X = videos[['videoViewCount', 'likeCount', 'dislikeCount', 'commentCount', 'channelViewCount', 'subscriberCount', 'videoCount', 'positive_comments', 'negative_comments', 'neutral_comments', 'upload_date_timestamp', 'stock_date_timestamp', 'stock_price_before']].values
    y = videos[['stock_price_after']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # Create Regressors
    randomForrest = make_pipeline(
        MinMaxScaler(),
        RandomForestRegressor(min_samples_leaf=3)
    )

    linear = make_pipeline(
        MinMaxScaler(),
        LinearRegression()
    )

    KNN = make_pipeline(
        MinMaxScaler(),
        KNeighborsRegressor(n_neighbors=2, weights='distance', leaf_size=1, p=1)
    )

    MLP = make_pipeline(
        MinMaxScaler(),
        MLPRegressor(activation='tanh', solver='sgd', learning_rate='adaptive', max_iter=1500)
    )

    """
    # Test Parameters using GridSearchCV
    param_grid = { 
    'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu'], 
    'MLPRegressor__solver': ['lbfgs', 'sgd', 'adam'],
    'MLPRegressor__alpha': [0.00001, 0.0001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    
    CV_rfc = GridSearchCV(estimator=MLP, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train.ravel())
    print(CV_rfc.best_score_)
    print(CV_rfc.best_params_)
    """

    # Test Regressors for Best Test Score
    models = [randomForrest, linear, KNN, MLP]
    scoredf = pd.DataFrame()
    for i, m in enumerate(models):
        m.fit(X_train, y_train.ravel())
        scoredf[i] = [m.score(X_train, y_train), m.score(X_valid, y_valid)]

    scoredf.columns = ['Random Forest', 'Linear', 'KNN', 'MLP']
    scoredf.index = ['Train Score', 'Validation Score']
    bestmodel = scoredf.idxmax(axis=1)

    print(scoredf)
    print('Best Regressor:', bestmodel[1])
    print('Test Score:', scoredf[bestmodel[1]][0])
    print('Validation Score:', scoredf[bestmodel[1]][1])
    
    # Print Difference of Predictions and Actual Values
    predictions = pd.DataFrame(data={'prediction': models[scoredf.columns.get_loc(bestmodel[0])].predict(X_valid), 'actual': y_valid.reshape(len(y_valid), )})
    predictions.head(20).plot(kind='bar', figsize=(14, 7))
    plt.xlabel('First 20 Results')
    plt.ylabel('Stock Price')
    plt.title('Predicted vs Actual Prices')
    plt.savefig('ML Regressor Sample Results.png')
    
    # predictions['difference'] = abs(predictions['prediction'] - predictions['actual'])
    # print(predictions.sort_values(by='difference', ascending=False))