import streamlit as st
import google.generativeai as genai
import re
from config import API_KEY

# Configure Gemini API
genai.configure(api_key=API_KEY)

def generate_script(plot, genres, length, character_details, script_format):
    """Generates a movie script using Gemini LLM"""
    prompt = f"""
    Generate a {length} movie script in the {', '.join(genres)} genre(s).
    Format: {script_format} screenplay.
    Plot: {plot}
    Main Characters and Traits: {character_details}
    Include:
    - Title
    - Character Descriptions
    - Scene Breakdown
    - Unique Character-Specific Dialogues
    - Actions and Settings
    - Three Alternative Endings
    """
    model = genai.GenerativeModel("gemini-1.5-pro")  # Corrected model name
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)

def extract_endings(script_text):
    """Extracts three endings from the script"""
    endings = re.findall(r'ENDING:.*?(?=\n[A-Z]+:|\Z)', script_text, re.DOTALL)
    return endings[-3:] if len(endings) >= 3 else endings

def save_as_txt(text, filename="movie_script.txt"):
    """Saves the script as a plain text file"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    return filename

# Streamlit UI
st.title("🎬 Movie Script Generator")
st.markdown("Create a custom movie script using AI!")

# User Inputs
plot = st.text_area("Enter a short plot:")
genres = st.multiselect("Select Genres:", ["Action", "Comedy", "Horror", "Sci-Fi", "Drama", "Fantasy", "Thriller","Romance","Mystery","Adventure","Animation","Family","Historical","Biography"])
length = st.radio("Choose Script Length:", ["Short", "Long"])
script_format = st.selectbox("Choose Script Format:", ["Hollywood", "Bollywood", "Playwriting"])
character_details = st.text_area("Enter Character Names & Traits (e.g., John - Brave, Emily - Smart):")

if st.button("Generate Script 🎥"):
    if plot and genres and character_details:
        script = generate_script(plot, genres, length, character_details, script_format)
        st.text_area("Generated Script:", script, height=400)
        
        # Save script as .txt file
        txt_file = save_as_txt(script)
        
        # Provide download button for the text file
        with open(txt_file, "rb") as file:
            st.download_button("Download Full Script as TXT", file, file_name="movie_script.txt")

        endings = extract_endings(script)
        if endings:
            selected_ending = st.radio("Choose an Ending:", endings)
            
            if st.button("Download Selected Ending as TXT"):
                ending_txt_file = save_as_txt(selected_ending, "movie_script_ending.txt")
                with open(ending_txt_file, "rb") as file:
                    st.download_button("Download Ending as TXT", file, file_name="movie_script_ending.txt")
    else:
        st.warning("Please fill in all fields!")



        #2code#


import streamlit as st
import google.generativeai as genai
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import API_KEY
import pickle  # To store and load scripts

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Load embedding model (free)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  

# FAISS Setup
embedding_dim = 384  # MiniLM produces 384-dimensional vectors
index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity search
script_database = {}  # Store scripts with their IDs

# Function to generate embeddings
def get_embedding(text):
    return embedding_model.encode([text])[0]

# Function to retrieve similar scripts using FAISS
def retrieve_similar_scripts(plot):
    """Finds the most relevant script based on FAISS similarity search."""
    query_embedding = get_embedding(plot).astype(np.float32).reshape(1, -1)
    if index.ntotal == 0:
        return []  # No data yet

    _, indices = index.search(query_embedding, k=3)  # Get top 3 similar scripts
    return [script_database[idx] for idx in indices[0] if idx in script_database]

# Function to generate script with RAG
def generate_script(plot, genres, length, character_details, script_format):
    """Generates a movie script using Gemini LLM with RAG"""
    
    retrieved_scripts = retrieve_similar_scripts(plot)
    rag_context = "\n\n".join(retrieved_scripts) if retrieved_scripts else "No relevant scripts found."
    
    prompt = f"""
    You are a professional screenplay writer.
    
    Generate a {length} movie script in the {', '.join(genres)} genre(s).
    Format: {script_format} screenplay.
    Plot: {plot}
    Main Characters and Traits: {character_details}

    --- Retrieved Similar Scripts for Context ---
    {rag_context}

    Include:
    - Title
    - Character Descriptions
    - Scene Breakdown
    - Unique Character-Specific Dialogues
    - Actions and Settings
    - Three Alternative Endings
    """

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)

def extract_endings(script_text):
    """Extracts three endings from the script"""
    endings = re.findall(r'ENDING:.*?(?=\n[A-Z]+:|\Z)', script_text, re.DOTALL)
    return endings[-3:] if len(endings) >= 3 else endings

def save_as_txt(text, filename="movie_script.txt"):
    """Saves the script as a plain text file"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    return filename

def add_script_to_db(script):
    """Embeds and stores a script in FAISS for retrieval"""
    global script_database
    script_id = len(script_database)  # Unique ID for each script
    script_database[script_id] = script
    script_embedding = get_embedding(script).astype(np.float32)
    index.add(np.array([script_embedding]))  # Add to FAISS

# Streamlit UI
st.title("🎬 Movie Script Generator with Free RAG")
st.markdown("Create a custom movie script using AI and retrieved knowledge!")

# User Inputs
plot = st.text_area("Enter a short plot:")
genres = st.multiselect("Select Genres:", ["Action", "Comedy", "Horror", "Sci-Fi", "Drama", "Fantasy", "Thriller","Romance","Mystery","Adventure","Animation","Family","Historical","Biography"])
length = st.radio("Choose Script Length:", ["Short", "Long"])
script_format = st.selectbox("Choose Script Format:", ["Hollywood", "Bollywood", "Playwriting"])
character_details = st.text_area("Enter Character Names & Traits (e.g., John - Brave, Emily - Smart):")

if st.button("Generate Script 🎥"):
    if plot and genres and character_details:
        script = generate_script(plot, genres, length, character_details, script_format)
        st.text_area("Generated Script:", script, height=400)

        # Save and allow download
        txt_file = save_as_txt(script)
        with open(txt_file, "rb") as file:
            st.download_button("Download Full Script as TXT", file, file_name="movie_script.txt")

        endings = extract_endings(script)
        if endings:
            selected_ending = st.radio("Choose an Ending:", endings)
            if st.button("Download Selected Ending as TXT"):
                ending_txt_file = save_as_txt(selected_ending, "movie_script_ending.txt")
                with open(ending_txt_file, "rb") as file:
                    st.download_button("Download Ending as TXT", file, file_name="movie_script_ending.txt")
        
        # Store script in the database for future retrieval
        add_script_to_db(script)
    else:
        st.warning("Please fill in all fields!")

# Save database persistently
if st.button("Save Script Database"):
    with open("script_db.pkl", "wb") as db_file:
        pickle.dump(script_database, db_file)
    st.success("Script database saved!")