import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re

# No longer need to manually set the tesseract_cmd path!
# The server environment will handle this.

DATABASE_FILE = "players.csv"

# --- Database and OCR Functions (No changes here) ---

def init_csv(filename=DATABASE_FILE):
    if not os.path.exists(filename):
        # We assume the CSV will be in the repository, but can create a fallback
        st.info("Creating a fallback sample CSV file.")
        sample_data = {
            'roblox_username': ['Nethy', 'yio', 'SLAYER_ItzzRoBdabest', 'Stark', 'Ultra', 'Woofus', 'KING_Doge'],
            'discord_username': ['Nethy#1234', 'yio_discord#5678', 'Slayer#4444', 'Stark_D#9911', 'UltraGamer#3322', 'Woofus_#8888', 'DogeKing#7777']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(filename, index=False)

def get_player_from_csv(username, db_dataframe):
    result = db_dataframe[db_dataframe['roblox_username'].str.lower() == username.lower()]
    if not result.empty:
        return result['discord_username'].iloc[0]
    return None

def extract_text_from_image(image):
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        st.error(f"An error occurred during OCR processing on the server: {e}")
        return None

def parse_leaderboard_text(text):
    player_names = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        st.warning("Could not find 'Leaderboard' title. Results may be inaccurate.")
        relevant_lines = lines
    for line in relevant_lines:
        numbers_in_line = re.findall(r'\b\d{1,4}(?:,\d{3})*\b', line)
        if len(numbers_in_line) < 2:
            continue
        words = line.split()
        potential_name = ""
        for word in words:
            cleaned_word = re.sub(r'[^\w-]', '', word)
            if cleaned_word and not cleaned_word.isdigit():
                potential_name = cleaned_word
                break
        if potential_name and len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            player_names.append(potential_name)
    return list(dict.fromkeys(player_names))

# --- Streamlit GUI (No major changes) ---
st.set_page_config(page_title="Roblox Leaderboard Parser", layout="wide")
st.title("Web-Based Roblox Leaderboard Extractor")
st.write(f"This tool runs in the cloud! It extracts usernames from a screenshot and matches them with a Discord username from the `players.csv` database.")
init_csv() # In a real app, you would manage your CSV directly in the repo
uploaded_file = st.file_uploader("Upload a full or cropped screenshot of a leaderboard", type=["png", "jpg", "jpeg"])
# ... (rest of the GUI code is the same as the last version)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_column_width=True)
    if st.button("Find Players in Database"):
        # ... (processing logic remains identical)
        with st.spinner("Step 1/3: Analyzing screenshot and reading text..."):
            leaderboard_text = extract_text_from_image(image)
        if leaderboard_text is None or not leaderboard_text.strip():
            st.error("OCR could not detect any text.")
        else:
            st.subheader("Raw Text Extracted by OCR")
            st.text_area("This is all the text the program sees in your image:", leaderboard_text, height=200)
            with st.spinner("Step 2/3: Identifying usernames..."):
                found_names = parse_leaderboard_text(leaderboard_text)
            if not found_names:
                st.error("Could not identify any valid usernames.")
            else:
                st.subheader("Usernames Found in Leaderboard")
                st.dataframe(pd.DataFrame(found_names, columns=["Identified Roblox Usernames"]))
                try:
                    player_db_df = pd.read_csv(DATABASE_FILE)
                    matched_players = []
                    with st.spinner(f"Step 3/3: Searching for matches..."):
                        for name in found_names:
                            discord_username = get_player_from_csv(name, player_db_df)
                            if discord_username:
                                matched_players.append({'Roblox Username': name, 'Discord Username': discord_username})
                    st.subheader("✅ Cross-Referencing Complete")
                    if matched_players:
                        st.success(f"Found {len(matched_players)} matching players!")
                        st.dataframe(pd.DataFrame(matched_players))
                    else:
                        st.warning("No players from the leaderboard were found in the database.")
                except FileNotFoundError:
                    st.error(f"Database file '{DATABASE_FILE}' not found on the server.")
                except Exception as e:
                    st.error(f"An error occurred while reading the database: {e}")