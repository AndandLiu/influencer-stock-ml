import pandas as pd
import numpy as np
from googleapiclient.discovery import build
import datetime
from textblob import TextBlob
import sys
# pd.set_option("display.max_rows", None, "display.max_columns", None)

API_KEY = 'AIzaSyDdMXol7uE3mVfyX2KOw1Mu0hsGu-QsHjQ'
NUM_VIDEOS_PER_PAGE = 50
NUM_SEARCH_PAGES = 20

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_search_results(query):
	search_responses = []

	initial_search_response = youtube.search().list(
		q=query,
		part='id,snippet',
		maxResults=NUM_VIDEOS_PER_PAGE,
		order='viewCount',
		publishedAfter='2016-01-01T00:00:00Z' 
	).execute()

	next_page_token = initial_search_response['nextPageToken'] if 'nextPageToken' in initial_search_response else None

	counter = 0
	while next_page_token and counter < NUM_SEARCH_PAGES:
		next_search_response = youtube.search().list(
			q=query,
			part='id,snippet',
			maxResults=NUM_VIDEOS_PER_PAGE,
			pageToken=next_page_token,
			order='viewCount',
			publishedAfter='2016-01-01T00:00:00Z' 
		).execute()
		
		next_page_token = next_search_response['nextPageToken'] if 'nextPageToken' in next_search_response else None
		search_responses.append(next_search_response)
		counter += 1

	videos = []
	for search_response in search_responses:
		for response in search_response.get('items', []):
			if response['id']['kind'] == 'youtube#video':
				videos.append({
					'videoId': response['id']['videoId'],
					'title': response['snippet']['title'],
					'channelId': response['snippet']['channelId'],
					'channelTitle': response['snippet']['channelTitle'],
					'uploaded': response['snippet']['publishedAt'],
				})

	return pd.DataFrame(videos)


if __name__ == "__main__":
	query = sys.argv[1]
	output = f'{"_".join(query.split())}_search.csv'

	print(f'Getting results for query "{query}"')
	videos = get_search_results(query)
	# Only get unique videos
	videos = videos.drop_duplicates(subset='videoId', keep='first')

	# Save as csv
	print(f'Saving to output folder "{output}"')
	videos.to_csv(output, index=False)
	
   
