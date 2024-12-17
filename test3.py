import pandas as pd
import spacy
import requests
from bs4 import BeautifulSoup
from spacy import displacy
from textblob import TextBlob
import streamlit as st

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_rows", 200)

#_______________________________________________________________________________________________________________________
# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


#_______________________________________________________________________________________________________________________
# Function to get location from the children of a token
def get_children_ent(head):
    if head.children:
        for child in head.children:
            if child.ent_type_ == "LOC":
                return child
            else:
                # Recursively check the child's children
                loc = get_children_ent(child)
                if loc:  # If a location was found, return it
                    return loc
    return None  # Return None if no location is found

# Function to extract relationships between entities (Person and Location)
def extract_relationships(doc):
    relationships = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person = ent.text
            head = ent.root.head
            loc = get_children_ent(head)
            if loc:
                relationships.append((person, head.text, loc.text))
    return relationships


#_______________________________________________________________________________________________________________________
# Function to calculate precision, recall, and F1 score
def evaluate_relationships(extracted, ground_truth):
    true_positives = len(set(extracted) & set(ground_truth))
    precision = true_positives / len(extracted) if extracted else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


#_______________________________________________________________________________________________________________________
# Input content
st.title("RUFEERS")
st.subheader("INPUT")
content = st.text_area("Enter text for NLP analysis:")

# File uploader for text files
uploaded_file = st.file_uploader("Or choose a file from your device", type="txt")


#_______________________________________________________________________________________________________________________
# Fixed sidebar with group information
with st.sidebar:
    st.markdown("<h2 style='color: purple; font-weight: bold;'>Group Members</h2>", unsafe_allow_html=True)
    st.subheader("KHIZAR MEHMOOD\n  |  01-134212-074")
    st.subheader("MEENHOON KHAN\n   |  01-134212-085")
    st.subheader("FAHAD FARRUKH\n   |  01-134212-208")

    st.markdown("<h2 style='color: purple; font-weight: bold;'>Supervisor</h2>", unsafe_allow_html=True)
    st.subheader("Dr. Arif ur Rehman")

# If a file is uploaded, read its content
if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')

# Process the text only if there's content
if content:
    # Process the text
    doc = nlp(content)

    # Display entities
    st.subheader("Named Entities")
    entities = [(ent.text, ent.start_char, ent.end_char, ent.label_, ent.lemma_) for ent in doc.ents]
    entity_df = pd.DataFrame(entities, columns=['Text', 'Start', 'End', 'Type', 'Lemma'])
    st.dataframe(entity_df)

    # Extract and display relationships
    st.subheader("Entity Relationships")
    relationships = extract_relationships(doc)
    if relationships:
        for rel in relationships:
            st.write(f"{rel[0]} {rel[1]} {rel[2]}")  # Display "Entity1 Action Location"
    else:
        st.write("No relationships found.")

    # Example ground truth for evaluation (modify this with actual expected relationships)
    ground_truth = [("Khan", "arrested", "the High Court")]  # Modify according to your data

    # Evaluate extracted relationships
    # precision, recall, f1_score = evaluate_relationships(relationships, ground_truth)

    # Display evaluation metrics
    # st.subheader("Evaluation Metrics")
    # st.write(f"**Precision:** {precision:.2f}")
    # st.write(f"**Recall:** {recall:.2f}")
    # st.write(f"**F1 Score:** {f1_score:.2f}")


    #_______________________________________________________________________________________________________________________
    # Sentiment Analysis
    sentiment_score = analyze_sentiment(content)
    st.subheader("Sentiment Analysis")
    if sentiment_score > 0:
        st.write(f"**Positive Sentiment Score:** {sentiment_score:.2f}")
    elif sentiment_score < 0:
        st.write(f"**Negative Sentiment Score:** {sentiment_score:.2f}")
    else:
        st.write("**Neutral Sentiment**")

    
    #_______________________________________________________________________________________________________________________
    # Visualize named entities
    st.subheader("Entity Visualization")
    html = displacy.render(doc, style="ent", jupyter=False)
    st.components.v1.html(html, height=500)
    html2 = displacy.render(doc, style="dep")
    st.components.v1.html(html2, height=500, width = 500)

    
    #_______________________________________________________________________________________________________________________
    # Additional Feature: Web Scraping for Related News
    def fetch_related_news(query):
        try:
            url = f"https://news.google.com/search?q={query}"
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [h.text for h in soup.find_all('h3')]
            return headlines
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []

    if st.button("Fetch Related News"):
        st.write("Fetching news...")
        if not entity_df.empty:  # Check if entity_df has any entities
            # Fetch news for the first entity or modify to pick as needed
            query = entity_df['Text'].iloc[0]  # Get the first entity as the query
            related_news = fetch_related_news(query)
            
            if related_news:
                st.subheader("Related News Headlines")
                for news in related_news:
                    st.write(f"- {news}")
            else:
                st.write("No news found.")
        else:
            st.warning("No entities found to fetch news.")
else:
    st.warning("Please enter text or upload a file for analysis.")