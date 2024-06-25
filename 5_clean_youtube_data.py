import pandas as pd
import numpy as np
import matplotlib as plt
import sys
from datetime import datetime
# pd.set_option("display.max_rows", None, "display.max_columns", None)

API_KEY = 'AIzaSyDfnf5Ldpv7LNPdCki0w9aJfy8ngF1PAYA'
MIN_SENTIMENT_COUNT = 50
LIKE_DISLIKE_RATIO = 1
MIN_NUMB_VIEWS = 50000
MIN_SUBSCRIBER_COUNT = 10000
MIN_VIDEO_COUNT = 50

def upload_date_parser(date):
    date = date.split(" ")[0].split("T")[0]
    return datetime.strptime(date, '%Y-%m-%d')
 
# Get the stock price one week after the video was released and one day prior to the video release
# If the stock price doesnt exist for given day then take the previous day
def get_stock_prices(video, stock):
    # Sort by date ascending
    stock = stock.sort_values(by=['Date'], ascending=True)
    upload_date = video['Date']

    before_stock = stock[stock['Date'] < upload_date]
    before_stock['keep'] = before_stock['Date'].apply(lambda date: pd.Timedelta(upload_date - date).total_seconds() / 3600 < 24 * 7)
    before_stock = before_stock[before_stock['keep'] == True].drop(columns=['keep'])

    after_stock = stock[stock['Date'] > upload_date]    
    after_stock["keep"] = after_stock["Date"].apply(lambda date: pd.Timedelta(date - upload_date).total_seconds() / 3600 < 24 * 7)
    after_stock = after_stock[after_stock['keep'] == True].drop(columns=['keep'])

    # Take the next available day
    before_stock_price = before_stock['Close'].iloc[-1] if len(before_stock) else None
    before_stock_date = before_stock['Date'].iloc[-1] if len(before_stock) else None

    # Take the next available day
    after_stock_price = after_stock['Close'].iloc[-1] if len(after_stock) else None
    after_stock_date = after_stock['Date'].iloc[-1] if len(after_stock) else None

    return pd.Series([before_stock_price, before_stock_date, after_stock_price, after_stock_date])


# Filter videos with low views and comment counts
# Filter videos with bad like to dislike ratio
# Filter creaters with low subscribers or number of videos
def filter_videos(videos):
	videos = videos[(videos['videoViewCount'] > MIN_NUMB_VIEWS) 
			& (videos['likeCount'].div(videos['dislikeCount']) > LIKE_DISLIKE_RATIO) 
			& (videos['subscriberCount'] > MIN_SUBSCRIBER_COUNT) 
			& (videos['videoCount'] > MIN_VIDEO_COUNT)
	]
	return videos


if __name__ == "__main__":
    query = "_".join(sys.argv[1].split())
    videos = pd.read_csv(f'{query}_search.csv', parse_dates=['uploaded'], date_parser=upload_date_parser).rename(columns={'uploaded': 'Date'})
    channels = pd.read_csv(f'{query}_channels.csv')
    videoStats = pd.read_csv(f'{query}_video_stats.csv')
    comments = pd.read_csv(f'{query}_comments.csv')

    # Merge data sets
    videos = pd.merge(videos, videoStats, on='videoId')
    videos = pd.merge(videos, channels, on='channelId')
    stock = pd.read_csv(sys.argv[2], parse_dates=['Date'], index_col='stock')

    videos = filter_videos(videos)

    videos[['stock_price_before', 'stock_date_before', 'stock_price_after', 'stock_date_after']] = videos.apply(get_stock_prices, axis=1, args=(stock, ))
    videos = videos[(videos['stock_price_before'] != None) & (videos['stock_date_before'] != None) & (videos['stock_price_after'] != None) & (videos['stock_date_after'] != None)]
    videos = videos.drop_duplicates(subset='videoId', keep='first')
    videos.to_csv(f'{query}.csv', index=False)