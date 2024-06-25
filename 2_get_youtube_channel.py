import pandas as pd
import numpy as np
from googleapiclient.discovery import build
import datetime
from textblob import TextBlob
import sys
# pd.set_option("display.max_rows", None, "display.max_columns", None)

API_KEY = 'AIzaSyDdMXol7uE3mVfyX2KOw1Mu0hsGu-QsHjQ'

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_stats(channelIds):
    ids = ",".join(channelIds)
    channel_response = youtube.channels().list(
        part='statistics',
        id=ids
    ).execute()

    channels = []
    for response in channel_response.get('items', []):
        if response['kind'] == 'youtube#channel':
            stats = response['statistics']
            channels.append({
                'channelId': response['id'] if 'id' in response else None,
                'channelViewCount': stats['viewCount'] if 'viewCount' in stats else None,
                'subscriberCount': stats['subscriberCount'] if 'subscriberCount' in stats else None,
                'videoCount': stats['videoCount'] if 'videoCount' in stats else None,
            })

    return pd.DataFrame(channels)

if __name__ == "__main__":
    query = "_".join(sys.argv[1].split())
    filename = f'{query}_search.csv'
    output = f'{query}_channels.csv'

    print(f'Getting channel information for file "{filename}"')

    videos = pd.read_csv(filename)

    # Split ids into chunks 20 chunks
    channelIdChunks = np.array_split(np.unique(videos['channelId'].to_numpy()), 20)

    channels = pd.DataFrame()

    for channelIds in channelIdChunks:
        channels = pd.concat([channels, get_channel_stats(channelIds)])

    # Filter none values
    channels = channels[(channels['channelId'].notnull()) 
                & (channels['channelViewCount'].notnull()) 
                & (channels['subscriberCount'].notnull()) 
                & (channels['videoCount'].notnull())]

    # Save to csv
    print(f'Saving to output folder "{output}"')
    channels.to_csv(output, index=False)
