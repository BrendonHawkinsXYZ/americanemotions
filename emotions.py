import os
import openai
from datetime import datetime
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import time
import re
import random

# Initialize a PyTrends session
pytrends = TrendReq()

# Fetch daily trends for 'United States'
daily_trends = pytrends.trending_searches(pn='united_states')
search_term_column = daily_trends.columns[0]
daily_trends['Date'] = datetime.now().strftime('%Y-%m-%d')

# Define the 'terms' column as the search term column
daily_trends.rename(columns={search_term_column: 'terms'}, inplace=True)

# OpenAI API key
openai.api_key = "Yours here"

# Function to infer emotion and color from search term using OpenAI API
def infer_emotion_and_color(term):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes search terms and assigns emotions along with RGB colors. Provide RGB as three separate values."},
            {"role": "user", "content": f"Analyze the emotion of the following search term: '{term}'. Provide a two word response: an emotion and a color. Your response should be: Emotion: , Color:."}
        ]
    )
    text = response.choices[0].message['content']
    try:
        emotion_match = re.search(r"emotion: ([a-zA-Z ]+)", text)
        color_match = re.findall(r'\d+', text)
        emotion = emotion_match.group(1).lower() if emotion_match else ""
        if len(color_match) == 3 and all(0 <= int(x) <= 255 for x in color_match):
            color_rgb = tuple(int(x) for x in color_match)
        else:
            color_rgb = tuple(random.randint(0, 255) for _ in range(3))
    except Exception as e:
        print(f"Error processing term '{term}': {e}")
        emotion = "error"
        color_rgb = tuple(random.randint(0, 255) for _ in range(3))

    return text, emotion, color_rgb

# Process terms and extract data
response_data = [infer_emotion_and_color(term) for term in daily_trends['terms']]
responses, emotions, colors = zip(*response_data)

daily_trends['Response'] = responses
daily_trends['Emotion'] = emotions
daily_trends['Color'] = colors

# File path
file_path = 'daily_trending_searches_new.csv'

# Check if the file exists and is not empty
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    # If file exists and is not empty, read it and append new data
    existing_data = pd.read_csv(file_path)
    # Ensure the column name for search terms is 'terms'
    existing_data.rename(columns={existing_data.columns[0]: 'terms'}, inplace=True)
    # Append new data to the existing CSV file
    daily_trends.to_csv(file_path, mode='a', header=False, index=False)
else:
    # If file does not exist or is empty, save new data
    daily_trends.to_csv(file_path, index=False)

# Function to create a blended gradient map in portrait orientation
def create_gradient_map(colors):
    width = 400  # Width is now the shorter side
    height = 500  # Height is the longer side, making the image portrait
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
    gradient_map.save(f'gradient_portrait_{datetime.now().strftime("%Y-%m-%d")}.png')

print(f"Data saved for {datetime.now().strftime('%Y-%m-%d')}")
