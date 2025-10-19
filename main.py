import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re
from difflib import SequenceMatcher

# ==============================================================================
# This app is designed for deployment on Streamlit Community Cloud.
# It can process MULTIPLE screenshots, deduplicate player names, and
# uses a HYBRID matching system for maximum accuracy.
# ==============================================================================

DATABASE_FILE = "players.csv"
FUZZY_MATCH_THRESHOLD = 0.8

# --- Core Functions (No changes needed here) ---

def get_player_id_from_csv(username, db_dataframe):
    """[HYBRID] Fetches a player's Discord User ID using a robust two-step approach."""
    if db_dataframe.empty:
        return None

    # Step 1: High-confidence 'contains' search
    contains_result = db_dataframe[db_dataframe['roblox_username'].str.lower().str.contains(username.lower(), na=False)]
    if not contains_result.empty:
        return contains_result['discord_userid'].iloc[0]

    # Step 2: Fuzzy search fallback for misspellings
    best_match_score = 0.0
    best_match_id = None
    for index, row in db_dataframe.iterrows():
        db_username = row['roblox_username']
        similarity = SequenceMatcher(None, username.lower(), db_username.lower()).ratio()
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_id = row['discord_userid']
    if best_match_score >= FUZZY_MATCH_THRESHOLD:
        return best_match_id
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
    """Parses raw OCR text to extract player names from within a Roblox leaderboard."""
    player_names = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        relevant_lines = lines

    for line in relevant_lines:
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
    return player_names # Return list with potential duplicates for now

def format_discord_report(matched_discord_ids):
    """Formats the list of matched Discord User IDs into a Discord markdown block."""
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

# --- Streamlit GUI (Updated for Multi-File Handling) ---
st.set_page_config(page_title="Multi-Screenshot Report Generator", layout="wide")
st.title("Multi-Screenshot Leaderboard Report Generator")
st.write(f"Upload **one or more Roblox Leaderboard screenshots**. The app will combine the results, find unique players, and generate a single formatted event report.")

try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    if 'roblox_username' not in player_db_df.columns or 'discord_userid' not in player_db_df.columns:
        st.error(f"Error: Your '{DATABASE_FILE}' must contain 'roblox_username' and 'discord_userid' columns.")
        player_db_df = pd.DataFrame()
except FileNotFoundError:
    st.error(f"Error: The database file '{DATABASE_FILE}' was not found. Please upload it to your GitHub repository.")
    player_db_df = pd.DataFrame()

# --- NEW: Multi-file uploader ---
uploaded_files = st.file_uploader(
    "Upload one or more leaderboard screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("---")
    st.write("### Uploaded Screenshots:")
    # Display the images in columns for a neat layout
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i]:
            st.image(uploaded_file, caption=f"Image {i+1}", use_column_width=True)

    if st.button("Generate Discord Report from All Screenshots"):
        if player_db_df.empty:
            st.warning(f"Cannot process because the '{DATABASE_FILE}' file is missing, empty, or has incorrect columns.")
        else:
            all_extracted_names = []
            
            with st.spinner(f"Step 1/3: Analyzing {len(uploaded_files)} screenshot(s)..."):
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    ocr_text = extract_text_from_image(image)
                    if ocr_text:
                        names_from_image = parse_leaderboard_text(ocr_text)
                        all_extracted_names.extend(names_from_image)

            if not all_extracted_names:
                st.error("OCR could not detect any text in any of the uploaded images.")
            else:
                # --- NEW: Deduplication Step ---
                # This handles the overlapping screenshots automatically
                unique_names = list(dict.fromkeys(all_extracted_names))

                with st.spinner("Step 2/3: Finding unique players and matching with database..."):
                    matched_discord_ids = []
                    for name in unique_names:
                        discord_id = get_player_id_from_csv(name, player_db_df)
                        if discord_id:
                            matched_discord_ids.append(str(discord_id))
                    
                    matched_discord_ids = list(dict.fromkeys(matched_discord_ids))

                    with st.spinner("Step 3/3: Building final report..."):
                        discord_output = format_discord_report(matched_discord_ids)

                        st.subheader("✅ Your Combined Discord Report is Ready!")
                        st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
                        st.code(discord_output, language='markdown')

                        with st.expander("See processing details for all screenshots"):
                            st.write("**Unique Player Names Found (Combined & Deduplicated):**")
                            st.dataframe(pd.DataFrame(unique_names, columns=["Identified Roblox Usernames"]))
                            st.write("**Final Matched Discord User IDs:**")
                            if matched_discord_ids:
                                st.dataframe(pd.DataFrame(matched_discord_ids, columns=["Found Discord User IDs"]))
                            else:
                                st.warning(f"No players from any screenshot were found in {DATABASE_FILE}.")
