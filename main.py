import streamlit as st
import pandas as pd
from PIL import Image
import os
import re
from difflib import SequenceMatcher
import base64
import easyocr  # Using a neural network-based OCR library
import numpy as np

# ==============================================================================
# This is the definitive final version of the application.
# It uses a Neural Network (EasyOCR) for superior character recognition and
# a coordinate-based parser for perfect, column-wise data extraction.
# ==============================================================================

# --- Robust Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, "players.csv")
LOGO_FILE = os.path.join(SCRIPT_DIR, "logo.png")

FUZZY_MATCH_THRESHOLD = 0.8

# --- Global EasyOCR Reader Initialization ---
# This initializes the OCR model once and reuses it, which is much more efficient.
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False) # Use gpu=True if you have a compatible GPU

reader = load_ocr_model()

# --- Core Functions ---

def get_player_id_from_csv(username, db_dataframe):
    """[HYBRID] Fetches a player's Discord User ID using a robust two-step approach."""
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

def extract_text_from_image_nn(image):
    """[NEURAL NETWORK] Extracts text and its coordinates from an image using EasyOCR."""
    try:
        # Convert PIL image to a NumPy array, which is what EasyOCR needs
        image_np = np.array(image)
        # The 'detail=1' argument ensures we get coordinates, confidence, etc.
        result = reader.readtext(image_np, detail=1)
        return result
    except Exception as e:
        st.error(f"An error occurred during Neural Network OCR processing: {e}")
        return None

def parse_leaderboard_nn(ocr_results):
    """
    [COORDINATE-BASED PARSING] Groups OCR results into rows and columns based on their
    X and Y coordinates for perfect, column-wise separation.
    """
    if not ocr_results:
        return []

    # Group results into rows based on vertical position
    rows = {}
    for (bbox, text, prob) in ocr_results:
        # Get the vertical center of the bounding box
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        
        # Find a nearby row to group with
        found_row = False
        for row_y in rows.keys():
            if abs(y_center - row_y) < 20: # Tolerance for vertical alignment
                rows[row_y].append((bbox[0][0], text)) # Store x-coordinate and text
                found_row = True
                break
        
        if not found_row:
            rows[y_center] = [(bbox[0][0], text)]

    # Process each row to extract name and stats
    parsed_data = []
    for row_y in sorted(rows.keys()):
        # Sort items in the row by their horizontal position (left to right)
        sorted_row = sorted(rows[row_y], key=lambda item: item[0])
        
        # Combine text and numbers from the sorted row
        text_parts = [item[1] for item in sorted_row]
        
        # Separate name from stats
        name_parts = []
        stats = []
        for part in text_parts:
            cleaned_part = part.replace(',', '').replace('.', '')
            if cleaned_part.isdigit():
                stats.append(int(cleaned_part))
            else:
                name_parts.append(part)
        
        player_name = ' '.join(name_parts)
        # Clean up the name from any OCR artifacts or icons
        player_name = re.sub(r'[^\w\s-]', '', player_name).strip()

        if len(player_name) > 2 and stats and player_name.lower() not in ['people', 'score', 'win', 'coin']:
            parsed_data.append({'name': player_name, 'stats': stats})
            
    return parsed_data

def format_discord_report(matched_discord_ids):
    """Formats the final Discord markdown block."""
    report_lines = ["Event ID:", "Length:", "Host:", "Co-host:", "Attendees:"]
    if matched_discord_ids:
        for user_id in matched_discord_ids:
            report_lines.append(f"- <@{user_id}> | V")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
    report_lines.append("\nNote: ")
    return "\n".join(report_lines)

def set_watermark(file_path):
    """Sets a background watermark for the app."""
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
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Warning: Watermark file '{os.path.basename(file_path)}' not found.")

# --- Page Config ---
st.set_page_config(page_title="Blitzmarine Event-Logger", page_icon=LOGO_FILE, layout="wide")
set_watermark(LOGO_FILE)

# --- Session State Init ---
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.discord_output = ""
    st.session_state.unique_players = []
    st.session_state.players_with_scores = []

# --- Main App ---
st.title("Blitzmarine Event-Logger")
st.write("Upload **one or more Roblox Leaderboard screenshots**. A neural network will read the players and scores to generate a formatted event report.")

try:
    player_db_df = pd.read_csv(DATABASE_FILE)
    if 'roblox_username' not in player_db_df.columns or 'discord_userid' not in player_db_df.columns:
        st.error(f"Error: Your '{os.path.basename(DATABASE_FILE)}' must contain 'roblox_username' and 'discord_userid' columns.")
        player_db_df = pd.DataFrame()
except FileNotFoundError:
    st.error(f"Error: The database file '{os.path.basename(DATABASE_FILE)}' was not found.")
    player_db_df = pd.DataFrame()

uploaded_files = st.file_uploader(
    "Upload one or more leaderboard screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.expander(f"View the {len(uploaded_files)} uploaded screenshot(s)..."):
        for i, uploaded_file in enumerate(uploaded_files):
            st.image(uploaded_file, caption=f"Image {i+1}")
            if i < len(uploaded_files) - 1:
                st.divider()

    st.info("Please select which column represents the players' scores.")
    score_column_map = {"First Column": 0, "Second Column": 1, "Last Column": -1}
    selected_column_label = st.radio(
        "Which column is the score?",
        list(score_column_map.keys()),
        horizontal=True,
        index=2
    )
    score_column_index = score_column_map[selected_column_label]

    st.write("---")
    
    if st.button("Generate Discord Report from All Screenshots"):
        if player_db_df.empty:
            st.warning("Cannot process because the player database is missing or invalid.")
        else:
            all_parsed_data = []
            with st.spinner(f"Step 1/3: Analyzing {len(uploaded_files)} screenshot(s) with AI..."):
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file).convert("RGB")
                    ocr_results = extract_text_from_image_nn(image)
                    if ocr_results:
                        data_from_image = parse_leaderboard_nn(ocr_results)
                        all_parsed_data.extend(data_from_image)
            
            if not all_parsed_data:
                st.error("The AI could not detect any leaderboard data in the uploaded images.")
            else:
                unique_players_dict = {player['name']: player for player in reversed(all_parsed_data)}
                st.session_state.unique_players = list(unique_players_dict.values())
                
                with st.spinner("Step 2/3: Matching players and extracting scores..."):
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

if st.session_state.report_generated:
    st.write("---")
    st.subheader("✅ Your Discord Report is Ready!")
    st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
    st.code(st.session_state.discord_output, language='markdown')

    st.subheader("Player Scores")
    if st.session_state.players_with_scores:
        score_df = pd.DataFrame(st.session_state.players_with_scores)
        st.dataframe(score_df, use_container_width=True)
    else:
        st.warning("No players from the leaderboard were found in the database to display scores.")

    with st.expander("See raw processing details"):
        st.write("**Unique Player Data Found (Name & Stats):**")
        st.json(st.session_state.unique_players)

    if st.button("Start Over / Retry"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
