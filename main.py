import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re

# ==============================================================================
# This app is designed for deployment on Streamlit Community Cloud.
# It parses a ROBLOX LEADERBOARD screenshot and formats the output
# for a DISCORD EVENT LOG using User IDs for mentions.
# ==============================================================================

DATABASE_FILE = "players.csv"

# --- Database and OCR Functions ---

def get_player_id_from_csv(username, db_dataframe):
    """
    Fetches a player's Discord User ID from the provided DataFrame.
    Assumes the CSV has 'roblox_username' and 'discord_userid' columns.
    """
    if db_dataframe.empty:
        return None
    # .str.lower() makes the comparison case-insensitive
    result = db_dataframe[db_dataframe['roblox_username'].str.lower() == username.lower()]
    if not result.empty:
        # Return the value from the 'discord_userid' column
        return result['discord_userid'].iloc[0]
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
    [UPGRADED] Parses raw OCR text to extract player names from a Roblox leaderboard.
    This version is more robust and handles OCR errors from player icons.
    """
    player_names = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        st.warning("Could not find the 'Leaderboard' title. Results may be inaccurate.")
        relevant_lines = lines

    for line in relevant_lines:
        numbers_in_line = re.findall(r'\b\d{1,4}(?:,\d{3})*\b', line)
        if len(numbers_in_line) < 2:
            continue
        
        # New Logic: Find all non-numeric words and pick the best candidate (longest one)
        # This prevents icons (read as "B" or "@") from being chosen as the name.
        name_candidates = []
        for word in line.split():
            cleaned_word = re.sub(r'[^\w-]', '', word)
            if cleaned_word and not cleaned_word.isdigit():
                name_candidates.append(cleaned_word)

        if not name_candidates:
            continue
            
        # The longest word is the most likely username
        potential_name = max(name_candidates, key=len)
        
        if len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            player_names.append(potential_name)
            
    return list(dict.fromkeys(player_names))

def format_discord_report(matched_discord_ids):
    """
    [UPGRADED] Formats the list of matched Discord User IDs into a Discord markdown block.
    Uses the correct <@UserID> syntax for mentions.
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
            # This format creates a clickable mention in Discord
            report_lines.append(f"- <@{user_id}>")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
        
    return "\n".join(report_lines)

# --- Streamlit GUI ---
st.set_page_config(page_title="Leaderboard to Discord Report Generator", layout="wide")
st.title("Leaderboard to Discord Report Generator")
st.write(f"Upload a **Roblox Leaderboard screenshot**. The app will find players, look up their **Discord User ID** from `{DATABASE_FILE}`, and generate a formatted event report.")

# Load the player database at the start
try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    # Check if the required columns exist
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
                                # Ensure it's treated as a string to avoid formatting issues
                                matched_discord_ids.append(str(discord_id))
                        
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
