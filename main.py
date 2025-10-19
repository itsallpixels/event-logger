import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re
from difflib import SequenceMatcher

# ==============================================================================
# This app is designed for deployment on Streamlit Community Cloud.
# It uses Fuzzy String Matching for superior accuracy against OCR errors.
# ==============================================================================

DATABASE_FILE = "players.csv"
# The similarity threshold for a fuzzy match (0.8 = 80% similar)
# This is tunable. Higher is stricter, lower is more lenient.
MATCH_THRESHOLD = 0.8

# --- Database and OCR Functions ---

def get_player_id_from_csv(username, db_dataframe):
    """
    [UPGRADED] Fetches a player's Discord User ID using Fuzzy String Matching.
    This finds the best match in the database and accepts it if the similarity
    is above the MATCH_THRESHOLD.
    """
    if db_dataframe.empty:
        return None
    
    best_match_score = 0.0
    best_match_id = None
    
    # Iterate through every player in the database to find the best possible match
    for index, row in db_dataframe.iterrows():
        db_username = row['roblox_username']
        
        # Calculate the similarity ratio between the OCR'd name and the database name
        similarity = SequenceMatcher(None, username.lower(), db_username.lower()).ratio()
        
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_id = row['discord_userid']
            
    # If the best match we found is good enough, return the ID
    if best_match_score >= MATCH_THRESHOLD:
        return best_match_id
        
    # Otherwise, we conclude there was no confident match
    return None

def extract_text_from_image(image):
    """Extracts text from an image using Pytesseract OCR."""
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        st.error(f"An error occurred during OCR processing on the server: {e}")
        return None

def parse_leaderboard_text(text):
    """
    Parses raw OCR text to extract player names from within a Roblox leaderboard.
    """
    player_names = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        # For cropped images that don't have the word "leaderboard"
        relevant_lines = lines

    for line in relevant_lines:
        # A valid player line should have at least one number
        if not any(char.isdigit() for char in line):
            continue
        
        name_candidates = []
        for word in line.split():
            cleaned_word = re.sub(r'[^\w-]', '', word).strip('-_')
            if cleaned_word and not cleaned_word.isdigit():
                name_candidates.append(cleaned_word)

        if not name_candidates:
            continue
            
        potential_name = max(name_candidates, key=len)
        
        if len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            player_names.append(potential_name)
            
    return list(dict.fromkeys(player_names))

def format_discord_report(matched_discord_ids):
    """
    Formats the list of matched Discord User IDs into a Discord markdown block.
    """
    report_lines = [
        "Event ID: [fill this in]",
        "Length: [fill this in]",
        "Host: [fill this in]",
        "Co-host: [fill this in]",
        "Attendees:",
    ]
    
    if matched_discord_ids:
        for user_id in matched_discord_ids:
            report_lines.append(f"- <@{user_id}>")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
        
    return "\n".join(report_lines)

# --- Streamlit GUI ---
st.set_page_config(page_title="Leaderboard to Discord Report Generator", layout="wide")
st.title("Leaderboard to Discord Report Generator")
st.write(f"Upload a **Roblox Leaderboard screenshot**. The app will find players, look up their **Discord User ID** from `{DATABASE_FILE}`, and generate a formatted event report.")

try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    if 'roblox_username' not in player_db_df.columns or 'discord_userid' not in player_db_df.columns:
        st.error(f"Error: Your '{DATABASE_FILE}' must contain the columns 'roblox_username' and 'discord_userid'.")
        player_db_df = pd.DataFrame()
except FileNotFoundError:
    st.error(f"Error: The database file '{DATABASE_FILE}' was not found in the GitHub repository. Please upload it.")
    player_db_df = pd.DataFrame()

uploaded_file = st.file_uploader("Upload a leaderboard screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_column_width=True)

    if st.button("Generate Discord Report"):
        if player_db_df.empty:
            st.warning(f"Cannot process because the '{DATABASE_FILE}' file is missing, empty, or has incorrect columns.")
        else:
            with st.spinner("Step 1/3: Reading leaderboard screenshot..."):
                ocr_text = extract_text_from_image(image)

            if ocr_text is None or not ocr_text.strip():
                st.error("OCR could not detect any text. Please try a clearer screenshot.")
            else:
                with st.spinner("Step 2/3: Finding players and matching with database..."):
                    extracted_names = parse_leaderboard_text(ocr_text)
                    
                    if not extracted_names:
                        st.error("Could not identify any valid player names from the leaderboard.")
                    else:
                        matched_discord_ids = []
                        for name in extracted_names:
                            discord_id = get_player_id_from_csv(name, player_db_df)
                            if discord_id:
                                matched_discord_ids.append(str(discord_id))
                        
                        matched_discord_ids = list(dict.fromkeys(matched_discord_ids))

                        with st.spinner("Step 3/3: Building report..."):
                            discord_output = format_discord_report(matched_discord_ids)

                            st.subheader("✅ Your Discord Report is Ready!")
                            st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
                            st.code(discord_output, language='markdown')

                            with st.expander("See processing details"):
                                st.write("**Player Names Found in Screenshot:**")
                                st.dataframe(pd.DataFrame(extracted_names, columns=["Identified Roblox Usernames"]))
                                st.write("**Matched Discord User IDs:**")
                                if matched_discord_ids:
                                    st.dataframe(pd.DataFrame(matched_discord_ids, columns=["Found Discord User IDs"]))
                                else:
                                    st.warning(f"No players from this leaderboard were found in {DATABASE_FILE}.")
