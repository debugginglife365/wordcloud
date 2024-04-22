import streamlit as st
import pandas as pd
import os
import requests
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from langdetect import detect, LangDetectException
import matplotlib.font_manager as fm

nltk.download('punkt')

# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Initialize session state variables
if 'reset' not in st.session_state:
    st.session_state['reset'] = False
if 'date_range' not in st.session_state:
    st.session_state['date_range'] = None
if 'language_filter' not in st.session_state:
    st.session_state['language_filter'] = []
if 'string_filter' not in st.session_state:
    st.session_state['string_filter'] = []
if 'category_filter' not in st.session_state:
    st.session_state['category_filter'] = []
if 'assignee_filter' not in st.session_state:
    st.session_state['assignee_filter'] = []
if 'keywords' not in st.session_state:
    st.session_state['keywords'] = ""

# URL to the font file on GitHub
font_url = 'https://github.com/debugginglife365/Korean_fonts/blob/main/SeoulNamsanvert.ttf?raw=true'
font_path = 'SeoulNamsanvert.ttf'  # Local path where the font will be saved

# Download the font if it's not already downloaded
if not os.path.exists(font_path):
    response = requests.get(font_url)
    if response.status_code == 200:
        with open(font_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error("Failed to download the font file.")

# Initialize Okt
okt = Okt()

# Define stopwords
stopwords = ['만', '발', '좀', '수', '터', '제', '달', '칼', '요', '사려', '안', '소', '를', '개', '저', '왜', '나', '역', '퍼', '방', '트', '욥', '무']

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"


def load_data(uploaded_file):
    with st.spinner('Loading data...'):  # Show a spinner until the data is loaded
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if file_extension.lower() == '.xlsx':
                df = pd.read_excel(uploaded_file, parse_dates=['Date'])
            elif file_extension.lower() == '.csv':
                df = pd.read_csv(uploaded_file, parse_dates=['Date'])
            else:
                st.error("Unsupported file format.")
                return None

            df['Language'] = df['Contents'].apply(detect_language)
            df.sort_values(by='String', ascending=False, inplace=True)
            return df
        except Exception as e:
            st.error(f"Failed to decode the file due to: {e}")
            return None

def generate_word_cloud(data):
    with st.spinner('Generating word cloud...'):
        full_text = []
        for comment in data['Contents']:
            if data['Language'].iloc[0] == 'ko':
                extracted_nouns = okt.nouns(comment)
                filtered_nouns = [noun for noun in extracted_nouns if noun not in stopwords]
                full_text.extend(filtered_nouns)
            else:
                english_words = nltk.word_tokenize(comment)
                filtered_words = [word for word in english_words if word.lower() not in stopwords and word.isalpha()]
                full_text.extend(filtered_words)

        noun_counts = Counter(full_text)
        wordcloud = WordCloud(font_path=font_path, background_color='white', colormap='Pastel1',
                              width=800, height=800, max_words=200).generate_from_frequencies(noun_counts)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    st.success('Completed generating word cloud!')

def generate_bar_chart(data):
    with st.spinner('Generating bar chart...'):
        # Load Korean font
        font_path = 'SeoulNamsanvert.ttf'  # Ensure this path is correct
        prop = fm.FontProperties(fname=font_path)

        # Extract and filter nouns
        nouns = []
        for comment in data['Contents']:  # Adjust column name if necessary
            nouns.extend(okt.nouns(comment))
        
        filtered_nouns = []
        filtered_counts = []
        for noun, count in Counter(nouns).most_common(20):
            if noun not in stopwords:
                filtered_nouns.insert(0, noun)
                filtered_counts.insert(0, count)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(filtered_nouns, filtered_counts, color='#0c343d')
        plt.xlabel('개수', fontproperties=prop)
        plt.ylabel('키워드', fontproperties=prop)
        plt.title('Top 20 키워드', fontproperties=prop)
        plt.xticks(rotation=45, fontproperties=prop)
        plt.yticks(fontproperties=prop)
        for i, count in enumerate(filtered_counts):
            plt.text(count, i, str(count), ha='left', va='center', fontproperties=prop)
        st.pyplot(plt.gcf())
    st.success('Completed generating bar chart!')

st.title('Wordcloud Application for Multilingual Text')

# Sample data URL on GitHub
sample_data_url = 'https://github.com/debugginglife365/wordcloud/blob/main/sample.xlsx?raw=true'

# Download sample data button
response = requests.get(sample_data_url)
if response.status_code == 200:
    data = response.content
    st.download_button(
        label='Download Sample Data',
        data=data,
        file_name='sample_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.error("Failed to download the file.")



col1, col_spacer, col2 = st.columns([1.5, 0.1, 3])


with col1:
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None and 'Contents' in data.columns and 'Date' in data.columns:
            if data['Date'].notnull().all():
                min_date = data['Date'].min().to_pydatetime()
                max_date = data['Date'].max().to_pydatetime()

                if st.session_state['date_range'] is None or st.session_state['reset']:
                    st.session_state['date_range'] = (min_date, max_date)

                date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date,
                                       value=st.session_state['date_range'])
                st.session_state['date_range'] = date_range
                data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]

                language_filter = st.multiselect('Select Languages', options=data['Language'].unique(),
                                                 default=st.session_state['language_filter'])
                st.session_state['language_filter'] = language_filter

                string_filter = st.multiselect('Select String Values', options=data['String'].unique(),
                                               default=st.session_state['string_filter'])
                st.session_state['string_filter'] = string_filter

                category_filter = st.multiselect('Select Categories', options=data['Category'].unique(),
                                                 default=st.session_state['category_filter'])
                st.session_state['category_filter'] = category_filter

                assignee_filter = st.multiselect('Select Assignees', options=data['Assignee'].unique(),
                                                 default=st.session_state['assignee_filter'])
                st.session_state['assignee_filter'] = assignee_filter

                keywords_input = st.text_input("Enter keywords (comma separated)", value=st.session_state['keywords'])
                keywords = [keyword.strip().lower() for keyword in keywords_input.split(',') if keyword.strip()]
                st.session_state['keywords'] = keywords_input

                if keywords:
                    data = data[data['Contents'].str.contains('|'.join(keywords), case=False, na=False)]

                if language_filter:
                    data = data[data['Language'].isin(language_filter)]
                if string_filter:
                    data = data[data['String'].isin(string_filter)]
                if category_filter:
                    data = data[data['Category'].isin(category_filter)]
                if assignee_filter:
                    data = data[data['Assignee'].isin(assignee_filter)]

                if st.button("Reset Filters"):
                    st.session_state['reset'] = True
                    st.experimental_rerun()
                else:
                    st.session_state['reset'] = False
            else:
                st.error("Date column must contain valid dates.")
        else:
            st.error("Uploaded file does not contain the required columns: 'Contents' and/or 'Date'")
    else:
        st.write("Upload an Excel or CSV file to generate a word cloud.")

with col2:
    if uploaded_file is not None and not data.empty:
        visualization_type = st.selectbox('Choose Visualization Type', ['Word Cloud', 'Bar Chart'])
        if visualization_type == 'Word Cloud':
            generate_word_cloud(data)
        else:
            generate_bar_chart(data)
    else:
        st.write("No data available for the selected criteria.")
