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
# It includes an advanced Image Preprocessing pipeline for maximum OCR accuracy.
# ==============================================================================

# --- Robust Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, "players.csv")
LOGO_FILE = os.path.join(SCRIPT_DIR, "logo.png")

FUZZY_MATCH_THRESHOLD = 0.8

# --- NEW: Advanced Image Preprocessing Function ---
def preprocess_image_for_ocr(image):
    """
    Applies a series of image processing techniques to enhance the image for
    better OCR accuracy.
    """
    # Convert PIL Image to an OpenCV format (NumPy array)
    pil_image = image.convert('RGB')
    cv_image = np.array(pil_image)
    cv_image = cv_image[:, :, ::-1].copy() # Convert RGB to BGR

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # 2. Upscale the image for better detail (Tesseract works best at 300 DPI)
    # A scale factor of 2 is often a good starting point.
    width = int(gray.shape[1] * 2)
    height = int(gray.shape[0] * 2)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

    # 3. Apply adaptive thresholding to binarize the image (pure black & white)
    # This is excellent for handling variations in lighting.
    processed_image = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return processed_image

# --- Core Functions ---

def get_player_id_from_csv(username, db_dataframe):
    if db_dataframe.empty: return None
    contains_result = db_dataframe[db_dataframe['roblox_username'].str.lower().str.contains(username.lower(), na=False)]
    if not contains_result.empty: return contains_result['discord_userid'].iloc[0]
    best_match_score = 0.0
    best_match_id = None
    for index, row in db_dataframe.iterrows():
        db_username = row['roblox_username']
        similarity = SequenceMatcher(None, username.lower(), db_username.lower()).ratio()
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_id = row['discord_userid']
    if best_match_score >= FUZZY_MATCH_THRESHOLD: return best_match_id
    return None

def extract_text_from_image(image):
    """
    [UPGRADED] Extracts text from an image using Pytesseract OCR after preprocessing.
    """
    try:
        # Preprocess the image first for better accuracy
        processed_image = preprocess_image_for_ocr(image)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        return text
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}"); return None

def parse_leaderboard_text(text):
    """[DEFINITIVE UPGRADE] Parses raw OCR text with maximum accuracy for scores."""
    parsed_data = []
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if "leaderboard" in line.lower())
        relevant_lines = lines[start_index + 1:]
    except StopIteration:
        relevant_lines = lines
    for line in relevant_lines:
        number_matches = list(re.finditer(r'\b\d{1,3}(?:,\d{3})*|\d+\b', line))
        if not number_matches: continue
        first_number_start_index = number_matches[0].start()
        name_part = line[:first_number_start_index].strip()
        potential_name = re.sub(r'^\w\s+', '', name_part).strip()
        if not potential_name: potential_name = name_part
        stats = [int(match.group().replace(',', '')) for match in number_matches]
        if stats and len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            parsed_data.append({'name': potential_name, 'stats': stats})
    return parsed_data

def format_discord_report(matched_discord_ids):
    report_lines = ["Event ID:", "Length:", "Host:", "Co-host:", "Attendees:"]
    if matched_discord_ids:
        for user_id in matched_discord_ids:
            report_lines.append(f"- <@{user_id}> | V")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
    report_lines.append("\nNote: ")
    return "\n".join(report_lines)

def set_watermark(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{ background-color: #0F1116; }}
            .stApp::before {{
                content: ""; position: fixed; left: 0; top: 0; width: 100vw; height: 100vh;
                background-image: url("data:image/png;base64,{encoded_string}");
                background-position: center center; background-repeat: no-repeat;
                background-size: 40% auto; opacity: 0.1; z-index: -1;
            }}
            h1, h2, h3, h4, h5, h6 {{ color: #FFD700; }}
            .stButton>button {{ color: #0F1116; background-color: #FFD700; border-color: #FFD700; }}
            [data-testid="stFileUploader"] label {{ color: #FFD700; border-color: #FFD700; }}
            [data-testid="stExpander"] summary {{ color: #FFD700; }}
            [data-testid="stInfo"] {{ border-left-color: #FFD700; }}
            [data-testid="stSpinner"] > div {{ border-top-color: #FFD700; }}
            [data-testid="stProgressBar"] > div > div {{ background-image: linear-gradient(to right, #FFD700, #FFD700); }}
            [data-testid="stRadio"] label {{ color: #E0E0E0; }}
            [data-testid="stRadio"] label:hover {{ color: #FFD700 !important; }}
            [data-testid="stRadio"] input:checked + div {{ border-color: #FFD700 !important; }}
            [data-testid="stRadio"] input:checked + div::after {{ background-color: #FFD700 !important; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Warning: Watermark file '{os.path.basename(file_path)}' not found. Please upload it to your GitHub repository.")

# --- Page Config ---
st.set_page_config(page_title="Blitzmarine Event-Logger", page_icon=LOGO_FILE, layout="wide")
set_watermark(LOGO_FILE)

# --- Session State Init ---
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# --- Main App ---
st.title("Blitzmarine Event-Logger")
st.write("Upload **one or more Roblox Leaderboard screenshots**. The app will combine the results, find unique players, and generate a single formatted event report.")

try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    if 'roblox_username' not in player_db_df.columns or 'discord_userid' not in player_db_df.columns:
        st.error(f"Error: Your '{os.path.basename(DATABASE_FILE)}' must contain 'roblox_username' and 'discord_userid' columns.")
        player_db_df = pd.DataFrame()
except FileNotFoundError:
    st.error(f"Error: The database file '{os.path.basename(DATABASE_FILE)}' was not found in the GitHub repository. Please upload it.")
    player_db_df = pd.DataFrame()

uploaded_files = st.file_uploader("Upload one or more leaderboard screenshots", accept_multiple_files=True)

if uploaded_files:
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
        with st.spinner(f"Step 1/3: Preprocessing and analyzing {len(uploaded_files)} screenshot(s)..."):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                ocr_text = extract_text_from_image(image) # This now includes preprocessing
                if ocr_text:
                    all_parsed_data.extend(parse_leaderboard_text(ocr_text))
        
        if not all_parsed_data:
            st.error("OCR could not detect any text in any of the uploaded images.")
        else:
            unique_players_dict = {player['name']: player for player in reversed(all_parsed_data)}
            st.session_state.unique_players = list(unique_players_dict.values())
            with st.spinner("Step 2/3: Finding players, matching, and extracting scores..."):
                matched_ids = []
                players_with_scores = []
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
    st.write("---")
    st.subheader("✅ Your Combined Discord Report is Ready!")
    st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
    st.code(st.session_state.get('discord_output', ''))
    st.subheader("Player Scores")
    if st.session_state.get('players_with_scores'):
        score_df = pd.DataFrame(st.session_state.players_with_scores)
        st.dataframe(score_df, use_container_width=True)
    else:
        st.warning("No players from the leaderboard were found in the database to display scores.")
    with st.expander("See raw processing details"):
        st.write("**Unique Player Data Found (Name & Stats):**")
        st.json(st.session_state.get('unique_players', []))
    if st.button("Start Over / Retry"):
        st.session_state.report_generated = False
        st.session_state.pop('unique_players', None)
        st.session_state.pop('players_with_scores', None)
        st.session_state.pop('discord_output', None)
        st.session_state.pop('matched_ids', None)
        st.rerun()
