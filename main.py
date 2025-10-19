import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re

# ==============================================================================
# This app is designed for deployment on Streamlit Community Cloud.
# It parses a ROBLOX LEADERBOARD screenshot and formats the output
# for a DISCORD EVENT LOG.
# It reads player data from a public players.csv file in the same repository.
# ==============================================================================

DATABASE_FILE = "players.csv"

# --- Database and OCR Functions ---

def get_player_from_csv(username, db_dataframe):
    """Fetches a player's Discord username from the provided DataFrame, ignoring case."""
    if db_dataframe.empty:
        return None
    # .str.lower() makes the comparison case-insensitive
    result = db_dataframe[db_dataframe['roblox_username'].str.lower() == username.lower()]
    if not result.empty:
        return result['discord_username'].iloc[0]
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
        # Find the line containing "Leaderboard" to start parsing from there
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        # If "Leaderboard" isn't found, scan the whole text but warn the user
        st.warning("Could not find the 'Leaderboard' title. Results may be inaccurate.")
        relevant_lines = lines

    for line in relevant_lines:
        # A valid player line should have at least two numbers (score columns)
        numbers_in_line = re.findall(r'\b\d{1,4}(?:,\d{3})*\b', line)
        if len(numbers_in_line) < 2:
            continue
            
        words = line.split()
        potential_name = ""
        # The username is likely the first word that is not a number
        for word in words:
            cleaned_word = re.sub(r'[^\w-]', '', word) # Clean common OCR errors
            if cleaned_word and not cleaned_word.isdigit():
                potential_name = cleaned_word
                break # Found it, move to the next line
                
        # Final filter to avoid common header words and ensure it's a valid name
        if potential_name and len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            player_names.append(potential_name)
            
    return list(dict.fromkeys(player_names)) # Return unique names

def format_discord_report(matched_discord_users):
    """
    Formats the list of matched Discord usernames into a Discord markdown block.
    """
    report_lines = [
        "Event ID: [fill this in]",
        "Length: [fill this in]",
        "Host: [fill this in]",
        "Co-host: [fill this in]",
        "Attendees:",
    ]
    
    if matched_discord_users:
        for user in matched_discord_users:
            report_lines.append(f"- @{user}")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
        
    return "\n".join(report_lines)

# --- Streamlit GUI ---
st.set_page_config(page_title="Leaderboard to Discord Report Generator", layout="wide")
st.title("Leaderboard to Discord Report Generator")
st.write(f"Upload a **Roblox Leaderboard screenshot**. The app will find players, look them up in `{DATABASE_FILE}`, and generate a formatted event report.")

# Load the player database at the start to check if it exists
try:
    player_db_df = pd.read_csv(DATABASE_FILE)
except FileNotFoundError:
    st.error(f"Error: The database file '{DATABASE_FILE}' was not found in the GitHub repository. Please upload it.")
    player_db_df = pd.DataFrame() # Create an empty DataFrame to prevent other errors

uploaded_file = st.file_uploader("Upload a leaderboard screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_column_width=True)

    if st.button("Generate Discord Report"):
        # Check again in case the file was missing
        if player_db_df.empty:
            st.warning(f"Cannot process because the '{DATABASE_FILE}' file is missing or empty.")
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
                        matched_discord_users = []
                        for name in extracted_names:
                            discord_username = get_player_from_csv(name, player_db_df)
                            if discord_username:
                                matched_discord_users.append(discord_username)
                        
                        with st.spinner("Step 3/3: Building report..."):
                            discord_output = format_discord_report(matched_discord_users)

                            st.subheader("✅ Your Discord Report is Ready!")
                            st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
                            st.code(discord_output, language='markdown')

                            # Optional expander to show processing details
                            with st.expander("See processing details"):
                                st.write("**Player Names Found in Screenshot:**")
                                st.dataframe(pd.DataFrame(extracted_names, columns=["Identified Roblox Usernames"]))
                                st.write("**Matched Discord Usernames:**")
                                if matched_discord_users:
                                    st.dataframe(pd.DataFrame(matched_discord_users, columns=["Found Discord Usernames"]))
                                else:
                                    st.warning(f"No players from this leaderboard were found in {DATABASE_FILE}.")
