import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import openai
import pandas as pd
import random
import re
from PIL import Image, ImageDraw
import os

# Function to fetch and parse the RSS feed
def fetch_trends_rss():
    rss_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
    response = requests.get(rss_url)
    rss_content = response.content
    return rss_content

# Function to parse the RSS feed and extract trends
def parse_rss_feed(rss_content):
    root = ET.fromstring(rss_content)
    namespaces = {'ht': 'https://trends.google.com/trends/trendingsearches/daily'}
    search_trends = []

    for item in root.findall('./channel/item'):
        trend = {}
        trend['title'] = item.find('title').text if item.find('title') is not None else 'N/A'
        trend['traffic'] = item.find('ht:approx_traffic', namespaces).text if item.find('ht:approx_traffic', namespaces) is not None else 'N/A'
        trend['description'] = item.find('description').text if item.find('description') is not None else 'N/A'
        trend['link'] = item.find('link').text if item.find('link') is not None else 'N/A'
        trend['pubDate'] = item.find('pubDate').text if item.find('pubDate') is not None else 'N/A'
        trend['news_items'] = []

        for news_item in item.findall('ht:news_item', namespaces):
            news = {}
            news['title'] = news_item.find('ht:news_item_title', namespaces).text if news_item.find('ht:news_item_title', namespaces) is not None else 'N/A'
            news['snippet'] = news_item.find('ht:news_item_snippet', namespaces).text if news_item.find('ht:news_item_snippet', namespaces) is not None else 'N/A'
            news['url'] = news_item.find('ht:news_item_url', namespaces).text if news_item.find('ht:news_item_url', namespaces) is not None else 'N/A'
            news['source'] = news_item.find('ht:news_item_source', namespaces).text if news_item.find('ht:news_item_source', namespaces) is not None else 'N/A'
            trend['news_items'].append(news)

        search_trends.append(trend)

    return search_trends

# Function to infer emotion and color from search term using OpenAI API
def infer_emotion_and_color(trend):
    news_items_str = '\n'.join([f"- {news['title']} ({news['source']}): {news['snippet']}" for news in trend['news_items']])
    prompt = (
        f"Analyze the emotion of the following search term and its context:\n"
        f"Title: {trend['title']}\n"
        f"Traffic: {trend['traffic']}\n"
        f"Description: {trend['description']}\n"
        f"News Items:\n{news_items_str}\n\n"
        f"Provide a response with an emotion and a corresponding unique color in this format: Emotion: <emotion>, Color: RGB(<red>, <green>, <blue>). Avoid an overbearing use of the same color tone. "
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes search terms and their context to assign emotions and corresponding UNIQUE RGB colors. Aim to use a diverse emotional vocabulary and a BROAD DIVERSE and unique color palette and tone. Provide RGB values as three separate numbers."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message['content']
    try:
        emotion_match = re.search(r"Emotion: ([a-zA-Z ]+),", text)
        color_match = re.findall(r'\d+', text)

        emotion = emotion_match.group(1).strip().lower() if emotion_match else "unknown"

        if len(color_match) == 3 and all(0 <= int(x) <= 255 for x in color_match):
            color_rgb = tuple(int(x) for x in color_match)
        else:
            color_rgb = tuple(random.randint(0, 255) for _ in range(3))
    except Exception as e:
        print(f"Error processing trend '{trend['title']}': {e}")
        emotion = "error"
        color_rgb = tuple(random.randint(0, 255) for _ in range(3))

    return text, emotion, color_rgb

# Main script
if __name__ == "__main__":
    # OpenAI API key
    openai.api_key = "Yours Here"

    # Fetch and parse the latest RSS feed
    rss_content = fetch_trends_rss()
    search_trends = parse_rss_feed(rss_content)

    # Infer emotions and colors for each trend
    response_data = [infer_emotion_and_color(trend) for trend in search_trends]
    responses, emotions, colors = zip(*response_data)

    # Create a DataFrame to store the results
    df_trends = pd.DataFrame(search_trends)
    df_trends['Response'] = responses
    df_trends['Emotion'] = emotions
    df_trends['Color'] = colors

    # File path
    file_path = 'rss_trending_searches_new.csv'

    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # If file exists and is not empty, read it and append new data
        existing_data = pd.read_csv(file_path)
        # Ensure the column name for search terms is 'terms'
        existing_data.rename(columns={existing_data.columns[0]: 'terms'}, inplace=True)
        # Append new data to the existing CSV file
        df_trends.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # If file does not exist or is empty, save new data
        df_trends.to_csv(file_path, index=False)

    # Function to create a blended gradient map in portrait orientation
    def create_gradient_map(colors):
        width = 4000  # Width is now the shorter side
        height = 5000  # Height is the longer side, making the image portrait
        gradient = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(gradient)

        if len(colors) > 1:
            for i in range(len(colors) - 1):
                r1, g1, b1 = colors[i]
                r2, g2, b2 = colors[i + 1]
                start = i * (width // (len(colors) - 1))
                end = (i + 1) * (width // (len(colors) - 1)) if (i < len(colors) - 2) else width
                for x in range(start, end):
                    fraction = (x - start) / ((end - start) if (end - start) > 0 else 1)  # Prevent division by zero
                    r = r1 + (r2 - r1) * fraction
                    g = g1 + (g2 - g1) * fraction
                    b = b1 + (b2 - b1) * fraction
                    draw.line([(x, 0), (x, height)], fill=(int(r), int(g), int(b)))

        return gradient

    # Generate and save the gradient map
    gradient_map = create_gradient_map(colors)
    if gradient_map is not None:
        gradient_map.save(f'rss_portrait_{datetime.now().strftime("%Y-%m-%d")}.png')

    print(f"Data saved for {datetime.now().strftime('%Y-%m-%d')}")
