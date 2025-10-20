import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re
from difflib import SequenceMatcher
import base64
import cv2
import numpy as np

# ==============================================================================
# This is the definitive final version of the application.
# It uses positional OCR data analysis to solve the "split score" problem.
# ==============================================================================

# --- Robust Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, "players.csv")
LOGO_FILE = os.path.join(SCRIPT_DIR, "logo.png")

FUZZY_MATCH_THRESHOLD = 0.8

# --- Advanced Image Preprocessing Function ---
def preprocess_image_for_ocr(image):
    pil_image = image.convert('RGB')
    cv_image = np.array(pil_image)[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    width = int(gray.shape[1] * 2)
    height = int(gray.shape[0] * 2)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC) # Use Cubic for better quality
    # Apply a sharpening kernel to make text crisper
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    processed_image = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    return processed_image

# --- Core Functions ---
def get_player_id_from_csv(username, db_dataframe):
    if db_dataframe.empty: return None
    contains_result = db_dataframe[db_dataframe['roblox_username'].str.lower().str.contains(username.lower(), regex=False, na=False)]
    if not contains_result.empty: return contains_result['discord_userid'].iloc[0]
    best_match_score = 0.0
    best_match_id = None
    for _, row in db_dataframe.iterrows():
        db_username = row['roblox_username']
        similarity = SequenceMatcher(None, username.lower(), db_username.lower()).ratio()
        if similarity > best_match_score: best_match_score, best_match_id = similarity, row['discord_userid']
    if best_match_score >= FUZZY_MATCH_THRESHOLD: return best_match_id
    return None

def extract_text_data_from_image(image):
    """
    [UPGRADED] Extracts detailed positional data from the image, not just text.
    """
    try:
        processed_image = preprocess_image_for_ocr(image)
        # PSM 6 is best for blocks of text. We will reconstruct layout manually.
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
        return data
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}"); return None

def parse_leaderboard_text(ocr_data):
    """
    [DEFINITIVE UPGRADE] Analyzes positional OCR data to intelligently group split numbers.
    """
    if not ocr_data: return []
    
    # Group words by line
    lines = {}
    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 30: # Confidence threshold
            (x, y, w, h, text) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i], ocr_data['text'][i])
            if text.strip():
                line_num = ocr_data['line_num'][i]
                if line_num not in lines: lines[line_num] = []
                lines[line_num].append({'text': text, 'left': x, 'width': w, 'conf': ocr_data['conf'][i]})
    
    parsed_data = []
    for line_num in lines:
        words = sorted(lines[line_num], key=lambda item: item['left'])
        
        # --- NEW: Intelligent Number Grouping Logic ---
        grouped_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            if current_word['text'].isdigit() and i + 1 < len(words):
                next_word = words[i+1]
                if next_word['text'].isdigit():
                    # Check the gap between the two numbers
                    gap = next_word['left'] - (current_word['left'] + current_word['width'])
                    # If gap is small (e.g., less than the width of one character), merge them
                    if gap < (current_word['width'] / len(current_word['text'])) * 1.5:
                        current_word['text'] += next_word['text']
                        current_word['width'] = (next_word['left'] + next_word['width']) - current_word['left']
                        i += 1 # Skip the next word since it has been merged
                        continue
            grouped_words.append(current_word)
            i += 1
        
        # Now parse the cleaned, grouped words
        name_parts = []
        stats = []
        found_first_number = False
        for word in grouped_words:
            if word['text'].isdigit() and not found_first_number:
                found_first_number = True
            
            if not found_first_number:
                name_parts.append(word['text'])
            else:
                # Try to convert to int, ignore if it fails (e.g. OCR error)
                try: stats.append(int(word['text'].replace(',', '')))
                except ValueError: pass
        
        potential_name = " ".join(name_parts).strip()
        
        if stats and len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team', 'leaderboard']:
            parsed_data.append({'name': potential_name, 'stats': stats})
            
    return parsed_data

def format_discord_report(matched_discord_ids):
    report_lines = ["Event ID:", "Length:", "Host:", "Co-host:", "Attendees:"]
    if matched_discord_ids:
        for user_id in matched_discord_ids: report_lines.append(f"- <@{user_id}> | V")
    else:
        report_lines.append("- (No attendees from the database)")
    report_lines.append("\nNote: ")
    return "\n".join(report_lines)

def set_watermark(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""<style>...</style>""", unsafe_allow_html=True) # CSS remains unchanged
    except FileNotFoundError:
        st.warning(f"Watermark file '{os.path.basename(file_path)}' not found.")

# --- Page Config & Main App ---
st.set_page_config(page_title="Blitzmarine Event-Logger", page_icon=LOGO_FILE, layout="wide")
set_watermark(LOGO_FILE)
if 'report_generated' not in st.session_state: st.session_state.report_generated = False
st.title("Blitzmarine Event-Logger")
st.write("Upload **one or more Roblox Leaderboard screenshots**...")

try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    if 'roblox_username' not in player_db_df.columns or 'discord_userid' not in player_db_df.columns:
        st.error(f"'{os.path.basename(DATABASE_FILE)}' must contain 'roblox_username' and 'discord_userid' columns.")
        player_db_df = pd.DataFrame()
except FileNotFoundError:
    st.error(f"'{os.path.basename(DATABASE_FILE)}' not found."); player_db_df = pd.DataFrame()

uploaded_files = st.file_uploader("Upload leaderboard screenshots", accept_multiple_files=True)

if uploaded_files:
    # ... (UI for displaying images and selecting score column remains the same)
    with st.expander(f"View the {len(uploaded_files)} uploaded screenshot(s)..."):
        for i, uploaded_file in enumerate(uploaded_files):
            st.image(uploaded_file, caption=f"Screenshot {i+1}", use_column_width=True)
    st.info("Please select which of the numeric columns represents the players' scores.")
    score_column_options = ("First Column", "Second Column", "Third Column")
    selected_column = st.radio("Which column is the score?", score_column_options, horizontal=True)
    score_column_index = score_column_options.index(selected_column)
    st.write("---")
    if st.button("Generate Discord Report from All Screenshots"):
        all_parsed_data = []
        raw_ocr_outputs = ""
        with st.spinner(f"Step 1/3: Analyzing layout of {len(uploaded_files)} screenshot(s)..."):
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                # This now returns detailed positional data
                ocr_data = extract_text_data_from_image(image)
                if ocr_data:
                    # For debugging, let's see the grouped words
                    raw_ocr_outputs += f"--- Analysis for Image {i+1} ---\n"
                    all_parsed_data.extend(parse_leaderboard_text(ocr_data))
        
        st.session_state.raw_ocr_output = raw_ocr_outputs # Store for debugging
        
        if not all_parsed_data:
            st.error("OCR could not detect any valid player data in the uploaded images.")
            st.session_state.report_generated = True
            st.rerun()
        else:
            # ... (Rest of the processing logic remains the same)
            unique_players_dict = {player['name']: player for player in reversed(all_parsed_data)}
            st.session_state.unique_players = list(unique_players_dict.values())
            with st.spinner("Step 2/3: Finding players, matching, and extracting scores..."):
                matched_ids, players_with_scores = [], []
                for player in st.session_state.unique_players:
                    discord_id = get_player_id_from_csv(player['name'], player_db_df)
                    if discord_id:
                        matched_ids.append(str(discord_id))
                        try:
                            score = player['stats'][score_column_index]
                            players_with_scores.append({'Roblox Username': player['name'], 'Discord User ID': str(discord_id), 'Score': score})
                        except IndexError:
                            players_with_scores.append({'Roblox Username': player['name'], 'Discord User ID': str(discord_id), 'Score': 'N/A'})
                st.session_state.matched_ids = list(dict.fromkeys(matched_ids))
                st.session_state.players_with_scores = players_with_scores
                with st.spinner("Step 3/3: Building final report..."):
                    st.session_state.discord_output = format_discord_report(st.session_state.matched_ids)
                    st.session_state.report_generated = True
                    st.rerun()

if st.session_state.report_generated:
    # ... (Result display remains the same)
    st.write("---")
    st.subheader("✅ Your Combined Discord Report is Ready!")
    st.info("Click the copy icon below, then paste into Discord.")
    st.code(st.session_state.get('discord_output', ''))
    st.subheader("Player Scores")
    if st.session_state.get('players_with_scores'):
        st.dataframe(pd.DataFrame(st.session_state.players_with_scores), use_container_width=True)
    else:
        if st.session_state.get('unique_players'):
            st.warning("No players from the leaderboard were found in the database to display scores.")
            
    with st.expander("See raw processing details"):
        st.write("**Final Parsed Player Data (Name & Stats):**"); st.json(st.session_state.get('unique_players', []))
        
    if st.button("Start Over / Retry"):
        st.session_state.report_generated = False
        st.session_state.pop('unique_players', None); st.session_state.pop('players_with_scores', None)
        st.session_state.pop('discord_output', None); st.session_state.pop('matched_ids', None)
        st.session_state.pop('raw_ocr_output', None)
        st.rerun()
