import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import chi2_contingency, normaltest, mannwhitneyu, levene, f_oneway
from math import floor

from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

OUTPUT_TEMPLATE = (
    "Dataset: {dataset}\n"
    "Postive comments normality p-value: {positive_comment_normality:.3g}\n"
    "Negative comments normality p-value: {negative_comment_normality:.3g}\n"
    "Neutral comments normality p-values: {neutral_comment_normality:.3g}\n"
    # "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def get_videos(folder):
    all_files = glob(folder + "/*.csv")

    videos = pd.DataFrame()
    for fileName in all_files:
        df = pd.read_csv(fileName, parse_dates=['Date', 'stock_date_before', 'stock_date_after'])
        videos = pd.concat([videos, df])

    # Get unique videos
    videos = videos.drop_duplicates(subset='videoId', keep='first').dropna()
    videos = videos.drop(columns=['title', 'channelTitle', 'commentCount'])
    videos = videos[(videos['positive_comments'] > 0) & (videos['neutral_comments'] > 0) & (videos['negative_comments'] > 0)]
    videos['abs_difference'] = videos.apply(get_abs_stock_difference, axis=1)
    videos['difference'] = videos.apply(get_difference, axis=1)
    videos['comment_difference'] = videos.apply(get_comment_difference, axis=1)
    return videos


def test_equal_variance(column1, column2):
    return levene(column1, column2).pvalue


def test_transform_normality(name, data):
    return f'''
    {name}:
    Postive comments normality p-value: {normaltest(np.log(data['positive_comments'])).pvalue}
    Negative comments normality p-value: {normaltest(np.log(data['negative_comments'])).pvalue}
    Neutral comments normality p-values: {normaltest(np.log(data['neutral_comments'])).pvalue}
    '''

def test_normality(name, data):
    return f'''
    {name}:
    Postive comments normality p-value: {normaltest(data['positive_comments']).pvalue}
    Negative comments normality p-value: {normaltest(data['negative_comments']).pvalue}
    Neutral comments normality p-values: {normaltest(data['neutral_comments']).pvalue}
    '''

# this should be incorrect since those input data should be independent
# people will usually gives either neutral, positive or negative but not all.
'''
def mann_whitney_u_test(data):
    return mannwhitneyu(data['positive_comments'], data['neutral_comments']).pvalue
'''

def anova(data):
    anova = f_oneway(data['positive_comments', data['neutral_comments'], data['negative_comments']])
    print(anova)
    print(anova.pvalue)

def chi_square_test(google, apple, microsoft):
    contingency = [
        [google['positive_comments'].sum(), google['neutral_comments'].sum(), google['negative_comments'].sum(), google['likeCount'].sum(), google['dislikeCount'].sum()],
        [apple['positive_comments'].sum(), apple['neutral_comments'].sum(), apple['negative_comments'].sum(), apple['likeCount'].sum(), apple['dislikeCount'].sum()],
        [microsoft['positive_comments'].sum(), microsoft['neutral_comments'].sum(), microsoft['negative_comments'].sum(), microsoft['likeCount'].sum(), microsoft['dislikeCount'].sum()]
    ]
    chi2, p, dof, expected = chi2_contingency(contingency)
    return p


def get_abs_stock_difference(data):
    return abs(data.stock_price_after - data.stock_price_before)

def get_difference(data):
    return (data['stock_price_after'] - data['stock_price_before'])

def get_comment_difference(data):
    return (data['positive_comments'] - data['negative_comments'])

def get_avg_comment_sentiments(data):
    avg_positive = floor(data['positive_comments'].mean())
    avg_neutral = floor(data['neutral_comments'].mean())
    avg_negative = floor(data['negative_comments'].mean())

    return [avg_positive, avg_neutral, avg_negative]


# https://stackoverflow.com/questions/47796264/function-to-create-grouped-bar-plot
def create_avg_comment_graph(google, apple, microsoft):
    google_data = get_avg_comment_sentiments(google)
    apple_data = get_avg_comment_sentiments(apple)
    microsoft_data = get_avg_comment_sentiments(microsoft)

    mean_data = pd.DataFrame([
        ['Positive', 'Google', google_data[0]], ['Neutral', 'Google', google_data[1]], ['Negative', 'Google', google_data[2]],
        ['Positive', 'Apple', apple_data[0]], ['Neutral', 'Apple', apple_data[1]], ['Negative', 'Apple', apple_data[2]],
        ['Positive', 'Microsoft', microsoft_data[0]], ['Neutral', 'Microsoft', microsoft_data[1]], ['Negative', 'Microsoft', microsoft_data[2]]
    ], columns=['labels', 'column', 'val'])

    mean_data.pivot("column", "labels", "val").plot(kind='bar', figsize=(12, 7))
    plt.xlabel('Company')
    plt.ylabel('Number of comments')
    plt.title('Average Comment Sentiments')
    plt.savefig('avg_comment_sentiment.png')
    plt.clf()

# this is a test on those comment normally distributed
# all fails, so no t-test can be apply, nor ANOVA - f_oneway
def normal_test_flow(df, name):
    list = ['positive_comments', 'neutral_comments', 'negative_comments', 'likeCount', 'dislikeCount']
    count = 0
    for item in list:
        df_norm_p = stats.normaltest(df[item]).pvalue
        if(df_norm_p<0.05):
            #print('{} DF on {} is NOT normally distributed, p value {}'.format(name, item, df_norm_p))
            count+=1
        else:
            print('{} DF on {} normally distributed, p value {}'.format(name, item, df_norm_p))

        df_norm_p = stats.normaltest(np.log(df[item])).pvalue
        if(df_norm_p<0.05):
            #print('{} DF on {} is NOT normally distributed, p value {}'.format(name, item, df_norm_p))
            count+=1
        else:
            print('{} DF on {} normally distributed, p value {}'.format(name, item, df_norm_p))

        # we can see a lot data are right skewed
        #plt.hist(df[item])
        #plt.hist(np.log(df[item])) # looks more normal, but still not pass normal test
        #plt.title(item + ' ' + name)
        #plt.show()
    print('{} of column factors on {} DF has failed the normal distribute test \n'.format(count, name))


def create_trend_graph(df, dataname, fraction):
    df = df.sort_values(by = ['Date'])
    filtered_1 = lowess(df['difference'], df['Date'], frac=fraction)
    plt.plot(df['Date'], filtered_1[:, 1], 'r-', linewidth=1)
    filtered_2 = lowess(df['comment_difference'], df['Date'], frac=fraction)
    plt.plot(df['Date'], filtered_2[:, 1], 'b-', linewidth=1)
    plt.legend(("difference = after price - before price", "trend = positive comment - negetive comment"))
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Trend of {}'.format(dataname))
    plt.savefig('trend_{}.png'.format(dataname))
    plt.clf()
    #plt.show()

# compare two dataset are having same size on comments and likes
def mannwhitney_utest(df1, df2, name1, name2):
    list = ['positive_comments', 'neutral_comments', 'negative_comments', 'likeCount', 'dislikeCount']
    count = 0
    for item in list:
        p = mannwhitneyu(df1[item], df2[item]).pvalue
        if(p<0.05):
            #print('{} and {} have same size of data on {}, p value {}'.format(name1, name2, item, p))
            count+=1
        else:
            print('One of {} or {} has a larger size data on {}, p value {}'.format(name1, name2, item, p))
    print('{} and {} have {} datas are on the same size. \n'.format(name1, name2, count))


if __name__ == "__main__":
    google = get_videos('google')
    apple = get_videos('apple')
    microsoft = get_videos('microsoft')

    # some assumption for viewer
    # like and comment people are more active stock purchaser, so we focus on these group of people, assume they will be more reactive to the price change
    normal_test_flow(google, 'google')
    normal_test_flow(apple, 'apple')
    normal_test_flow(microsoft, 'microsoft')

    print(f'Chi Square p value: {chi_square_test(google, apple, microsoft)} \n')
    create_avg_comment_graph(google, apple, microsoft)
    # print(test_normality('Google', google))
    # print(test_equal_variance(google['positive_comments'], google['negative_comments']))
    # print(test_transform_normality('Google Log', google))

    create_trend_graph(google, 'google', 0.1) # try to make less noise than 0.05
    create_trend_graph(apple, 'apple', 0.1) # too low will make image error plotting
    create_trend_graph(microsoft, 'microsoft', 0.1) # too low will make image error plotting

    x = mannwhitney_utest(google, apple, 'google', 'apple')
    x = mannwhitney_utest(google, microsoft, 'google', 'microsoft')
    x = mannwhitney_utest(apple, microsoft, 'apple', 'microsoft')
