import sys
import datetime
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from textblob import TextBlob


API_KEY = 'AIzaSyDdMXol7uE3mVfyX2KOw1Mu0hsGu-QsHjQ'
NUM_COMMENTS_PAGES = 10
NUM_COMMENTS_PER_PAGE = 100
MIN_COMMENT_COUNT = 100
MIN_SENTIMENT_COUNT = 50
MIN_VIEW_COUNT = 1000

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(id):
    comments_responses = []
    comments_response = youtube.commentThreads().list(
        part="snippet",
        maxResults=NUM_COMMENTS_PER_PAGE,
        videoId=id,
        order='relevance'
    ).execute()

    comments_responses.append(comments_response)

    next_page_token = comments_response['nextPageToken'] if 'nextPageToken' in comments_response else None

    counter = 0
    while next_page_token and counter < NUM_COMMENTS_PAGES:
        next_page_comments = youtube.commentThreads().list(
            part="snippet",
            videoId=id,
            maxResults=NUM_COMMENTS_PER_PAGE,
            pageToken=next_page_token,
            order='relevance'
        ).execute()
        next_page_token = next_page_comments['nextPageToken'] if 'nextPageToken' in next_page_comments else None
        comments_responses.append(next_page_comments)
        counter += 1

    comments = []
    for comments_response in comments_responses:
        for comment in comments_response.get('items', []):
            comments.append({
                "videoId": comment['snippet']['topLevelComment']['snippet']['videoId'],
                "commentId": comment['id'],
                "text": comment['snippet']['topLevelComment']['snippet']['textDisplay'],
                "likes": comment['snippet']['topLevelComment']['snippet']['likeCount'],
                "comment_date": comment['snippet']['topLevelComment']['snippet']['publishedAt'],
            })
    return pd.DataFrame(comments)


def keep_comment(comment_date, upload_date):
	return pd.Timedelta(comment_date - upload_date).total_seconds() / 3600 < 24 * 30
	

# Only keep comments that have been posted in the same month as the video upload
def filter_comments(comments, upload_date):
	# Fix dates
	comments['comment_date'] = pd.to_datetime(comments['comment_date'])

	comments['keep_comment'] = comments['comment_date'].apply(keep_comment, args=(upload_date, ))
	comments = comments[comments['keep_comment'] == True].drop(columns=['keep_comment'])

	return comments


def get_comment_sentiment(text):
	testimonial = TextBlob(text)
	if testimonial.sentiment.polarity < 0: 
		return 'NEGATIVE'
	elif testimonial.sentiment.polarity == 0: 
		return 'NEUTRAL'
	else: 
		return 'POSITIVE'


def get_comment_sentiment_counts(video):
    try:
        comments = filter_comments(get_comments(video['videoId']), video['uploaded'])
        comments['sentiment'] = comments['text'].apply(get_comment_sentiment)

        sentiment_values = comments['sentiment'].value_counts()

        print(sentiment_values)

        positive = sentiment_values['POSITIVE'] if 'POSITIVE' in sentiment_values else 0
        negative = sentiment_values['NEGATIVE'] if 'NEGATIVE' in sentiment_values else 0
        neutral = sentiment_values['NEUTRAL'] if 'NEUTRAL' in sentiment_values else 0
        
        return pd.Series([positive, negative, neutral])
    except:
        return pd.Series([0, 0, 0])
    

if __name__ == "__main__":
    query = "_".join(sys.argv[1].split())
    filename = f'{query}_search.csv'
    output = f'{query}_comments.csv'

    videos = pd.read_csv(f'{query}_search.csv', parse_dates=['uploaded'])
    videoStats = pd.read_csv(f'{query}_video_stats.csv')
    
    videos = pd.merge(videos, videoStats, on='videoId')
    videos = videos[(videos['commentCount'] > MIN_COMMENT_COUNT) & (videos['videoViewCount'] > MIN_VIEW_COUNT)]
    videos[['positive_comments', 'negative_comments', 'neutral_comments']] = videos.apply(get_comment_sentiment_counts, axis=1)
    videos = videos[videos.positive_comments + videos.negative_comments + videos.neutral_comments > MIN_SENTIMENT_COUNT]
    
    comments = videos[['videoId', 'positive_comments', 'negative_comments', 'neutral_comments']]
    print(f'Saving to directory "{output}"')
    comments.to_csv(output, index=False)

