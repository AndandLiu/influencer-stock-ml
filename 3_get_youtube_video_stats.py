import pandas as pd
import numpy as np
from googleapiclient.discovery import build
import datetime
from textblob import TextBlob
import sys
# pd.set_option("display.max_rows", None, "display.max_columns", None)

API_KEY = 'AIzaSyDdMXol7uE3mVfyX2KOw1Mu0hsGu-QsHjQ'

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_stats(videoIds):
    ids = ",".join(videoIds)
    stats_response = youtube.videos().list(
        part='statistics',
        id=ids
    ).execute()

    stats = []
    for response in stats_response.get('items', []):
        if response['kind'] == 'youtube#video':
            videoStats = response['statistics']
            stats.append({
                'videoId': response['id'] if 'id' in response else None,
                'videoViewCount': videoStats['viewCount'] if 'viewCount' in videoStats else None,
                'likeCount': videoStats['likeCount'] if 'likeCount' in videoStats else None,
                'dislikeCount': videoStats['dislikeCount'] if 'dislikeCount' in videoStats else None,
                'commentCount': videoStats['commentCount'] if 'commentCount' in videoStats else None
            })

    return pd.DataFrame(stats)


if __name__ == "__main__":
    query = "_".join(sys.argv[1].split())
    filename = f'{query}_search.csv'
    output = f'{query}_video_stats.csv'

    print(f'Getting video information for file "{filename}"')

    videos = pd.read_csv(filename)

    # Split ids into chunks 20 chunks
    videoIdChunks = np.array_split(np.unique(videos['videoId'].to_numpy()), 20)

    videoStats = pd.DataFrame()

    for videoIds in videoIdChunks:
        videoStats = pd.concat([videoStats, get_video_stats(videoIds)])

    videoStats = videoStats[(videoStats['videoId'].notnull()) 
                & (videoStats['videoViewCount'].notnull()) 
                & (videoStats['likeCount'].notnull()) 
                & (videoStats['dislikeCount'].notnull()) 
                & (videoStats['commentCount'].notnull())]

    print(videoStats)
    # Save to csv
    print(f'Saving to output folder "{output}"')
    videoStats.to_csv(output, index=False)
