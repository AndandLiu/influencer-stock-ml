import pandas as pd
import numpy as np
from glob import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

    # Get difference in stock price
    conditions = [(videos['stock_price_before'] < videos['stock_price_after']), (videos['stock_price_before'] > videos['stock_price_after'])]
    values = ['increase', 'decrease']
    videos['stock_price_difference'] = np.select(conditions, values)

    return fix_timestamps(videos)


if __name__ == "__main__":
    folder = sys.argv[1]

    videos = get_videos(folder)

    X = videos[['videoViewCount', 'likeCount', 'dislikeCount', 'commentCount', 'channelViewCount', 'subscriberCount', 'videoCount', 'positive_comments', 'negative_comments', 'neutral_comments', 'upload_date_timestamp', 'stock_date_timestamp', 'stock_price_before', 'stock_price_after']].values
    y = videos['stock_price_difference'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # Create Classifiers
    GB_model = GaussianNB()
    KNN_model = KNeighborsClassifier(n_neighbors=3, leaf_size=1)
    RF_model = RandomForestClassifier(n_estimators=25, criterion='entropy', min_samples_split=5, min_samples_leaf=3)
    MLP_model = MLPClassifier(activation='tanh', solver='sgd', learning_rate='constant', max_iter=500)

    """
    # Test Parameters using GridSearchCV
    param_grid = { 
    'activation': ['logistic','tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.00001, 0.0001, 0.01, 0.1],
    'learning_rate': ['constant','adaptive']
    }
    
    CV_rfc = GridSearchCV(estimator=MLP_model, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_score_)
    print(CV_rfc.best_params_)
    """

    # Test Classifiers for Best Test Score
    models = [GB_model, KNN_model, RF_model, MLP_model]
    scoredf = pd.DataFrame()
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        scoredf[i] = [m.score(X_train, y_train), m.score(X_valid, y_valid)]

    scoredf.columns = ['Gaussian Bayes', 'KNN', 'Random Forest', 'MLP']
    scoredf.index = ['Train Score', 'Validation Score']
    bestmodel = scoredf.idxmax(axis=1)
    
    print(scoredf)
    print('Best Classifier:', bestmodel[1])
    print('Test Score:', scoredf[bestmodel[1]][0])
    print('Validation Score:', scoredf[bestmodel[1]][1])
    
    # Print Number of Correct Predictions
    prediction = pd.DataFrame({'truth': y_valid, 'prediction': models[scoredf.columns.get_loc(bestmodel[0])].predict(X_valid)})
    correct = prediction[prediction['truth'] == prediction['prediction']]
    print(correct.shape[0], 'correct predictions out of', prediction.shape[0])