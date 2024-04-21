import streamlit as st
import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import os
import nltk
from langdetect import detect, LangDetectException

nltk.download('punkt')

# URL to the font file on GitHub
font_url = 'https://github.com/jaekookang/Korean-WordCloud/blob/master/SeoulNamsanvert.ttf?raw=true'
font_path = 'SeoulNamsanvert.ttf'  # Local path where the font will be saved

# Download the font if it's not already downloaded
if not os.path.exists(font_path):
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)

# Initialize Okt
okt = Okt()
stopwords = ['만', '발', '좀', '수', '터', '제', '달', '칼', '요', '사려', '안', '소', '를', '개', '저', '왜', '나', '역', '퍼', '방', '트', '욥', '무']

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, parse_dates=['Date'])
        df['Language'] = df['Contents'].apply(detect_language)
        df.sort_values(by='String', ascending=False, inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to decode the file due to: {e}")
        return None

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"

def generate_word_cloud(data):
    full_text = []
    for comment in data['Contents']:
        if data['Language'].iloc[0] == 'ko':  # Assumes the language has been filtered
            # Extract nouns from Korean text
            extracted_nouns = okt.nouns(comment)
            filtered_nouns = [noun for noun in extracted_nouns if noun not in stopwords]
            full_text.extend(filtered_nouns)
        else:
            # Tokenize English words
            english_words = nltk.word_tokenize(comment)
            filtered_words = [word for word in english_words if word.lower() not in stopwords and word.isalpha()]
            full_text.extend(filtered_words)

    noun_counts = Counter(full_text)
    wordcloud = WordCloud(font_path=font_path, background_color='white', colormap='Pastel1',
                          width=800, height=800, max_words=200).generate_from_frequencies(noun_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.title('Wordcloud Application for Multilingual Text')

uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None and 'Contents' in data.columns and 'Date' in data.columns:
        min_date = data['Date'].min().to_pydatetime()  # Convert to Python datetime
        max_date = data['Date'].max().to_pydatetime()  # Convert to Python datetime
        date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]
        
        # Continue with the rest of your filtering logic


        language_filter = st.multiselect('Select Languages', options=data['Language'].unique())
        if language_filter:
            data = data[data['Language'].isin(language_filter)]

        string_filter = st.multiselect('Select String Values', options=data['String'].unique())
        category_filter = st.multiselect('Select Categories', options=data['Category'].unique())
        assignee_filter = st.multiselect('Select Assignees', options=data['Assignee'].unique())

        # Apply other filters
        if string_filter:
            data = data[data['String'].isin(string_filter)]
        if category_filter:
            data = data[data['Category'].isin(category_filter)]
        if assignee_filter:
            data = data[data['Assignee'].isin(assignee_filter)]

        if not data.empty:
            st.write("Word Cloud Generation Complete!")
            generate_word_cloud(data)
        else:
            st.write("No data available for the selected criteria.")
    else:
        st.error("Uploaded file does not contain the required columns: 'Contents' and/or 'Date'")
else:
    st.write("Upload an Excel file to generate a word cloud.")
