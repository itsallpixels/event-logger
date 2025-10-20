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
# It uses positional OCR data to reconstruct the layout for maximum accuracy.
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
    width = int(gray.shape[1] * 2.5) # Upscale more for better detail
    height = int(gray.shape[0] * 2.5)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    processed_image = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    return processed_image

# --- NEW: Layout-Aware Text Reconstruction ---
def reconstruct_text_with_spacing(ocr_data):
    """
    Rebuilds the text from OCR data, inserting large spaces to simulate columns
    based on the horizontal position of words.
    """
    lines = {}
    for i in range(len(ocr_data['text'])):
        # We only care about words with some confidence
        if int(ocr_data['conf'][i]) > 40:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            text = ocr_data['text'][i]
            # Group words by their vertical position (line number)
            line_num = y // (h * 0.7) # Heuristic to group words on the same line
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append({'text': text, 'left': x, 'width': w})
    
    reconstructed_lines = []
    for line_num in sorted(lines.keys()):
        words = sorted(lines[line_num], key=lambda item: item['left'])
        line_text = ""
        for i, word in enumerate(words):
            line_text += word['text']
            if i < len(words) - 1:
                # Calculate the gap between this word and the next
                current_word_end = word['left'] + word['width']
                next_word_start = words[i+1]['left']
                gap = next_word_start - current_word_end
                # If the gap is large (e.g., more than the width of a couple of characters), treat it as a column break
                if gap > (word['width'] * 0.5):
                    line_text += "     " # Insert a large, consistent separator
                else:
                    line_text += " "
        reconstructed_lines.append(line_text)
    return "\n".join(reconstructed_lines)

# --- Core Functions ---
def get_player_id_from_csv(username, db_dataframe):
    if db_dataframe.empty: return None
    # Add regex=False to prevent crashes from special characters
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

def extract_text_from_image(image):
    """
    [UPGRADED] Extracts detailed positional data from the image.
    """
    try:
        processed_image = preprocess_image_for_ocr(image)
        # Ask for detailed data including word positions, not just plain text
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
        reconstructed_text = reconstruct_text_with_spacing(data)
        return reconstructed_text
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}"); return None

def parse_leaderboard_text(text):
    """
    [UPGRADED] Parses the pre-formatted text with high reliability.
    """
    parsed_data = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        relevant_lines = lines

    for line in relevant_lines:
        # Split the line by our large separator
        parts = re.split(r'\s{3,}', line) # Split by 3 or more spaces
        if len(parts) < 2: continue # A valid line must have at least a name and one stat column

        potential_name = parts[0].strip()
        # The rest of the parts are our stats
        stats_parts = parts[1:]
        
        try:
            # Extract numbers from the stats parts
            stats = [int(re.sub(r'[^\d]', '', part)) for part in stats_parts]
            if stats and len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
                parsed_data.append({'name': potential_name, 'stats': stats})
        except (ValueError, IndexError):
            continue
            
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
                # This now returns pre-formatted text
                ocr_text = extract_text_from_image(image)
                if ocr_text:
                    raw_ocr_outputs += f"--- Reconstructed Text for Image {i+1} ---\n{ocr_text}\n\n"
                    all_parsed_data.extend(parse_leaderboard_text(ocr_text))
        
        st.session_state.raw_ocr_output = raw_ocr_outputs
        
        if not all_parsed_data:
            st.error("OCR could not detect any text in any of the uploaded images.")
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
        if st.session_state.get('raw_ocr_output', '').strip():
            st.warning("No players from the leaderboard were found in the database to display scores.")
            
    with st.expander("See raw processing details"):
        st.write("**Reconstructed Text from OCR (Layout Aware):**")
        st.text_area("This is the text the app reconstructed by analyzing word positions.", st.session_state.get('raw_ocr_output', 'No text was detected.'), height=200)
        st.write("**Final Parsed Player Data (Name & Stats):**"); st.json(st.session_state.get('unique_players', []))
        
    if st.button("Start Over / Retry"):
        st.session_state.report_generated = False
        st.session_state.pop('unique_players', None); st.session_state.pop('players_with_scores', None)
        st.session_state.pop('discord_output', None); st.session_state.pop('matched_ids', None)
        st.session_state.pop('raw_ocr_output', None)
        st.rerun()
