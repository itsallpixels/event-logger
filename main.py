import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import os
import re
from difflib import SequenceMatcher
import base64

# ==============================================================================
# This is the final version of the application.
# It includes a robust full-page watermark logo, a custom theme, multi-file 
# support, and a retry mechanism.
# ==============================================================================

# --- Robust Path Configuration ---
# This ensures the script can always find its files, especially on Streamlit Cloud
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, "players.csv")
LOGO_FILE = os.path.join(SCRIPT_DIR, "logo.png")

FUZZY_MATCH_THRESHOLD = 0.8

# --- Core Functions (No changes) ---

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
        if not any(char.isdigit() for char in line): continue
        name_candidates = []
        for word in line.split():
            cleaned_word = re.sub(r'[^\w-]', '', word).strip('-_')
            if cleaned_word and not cleaned_word.isdigit(): name_candidates.append(cleaned_word)
        if not name_candidates: continue
        potential_name = max(name_candidates, key=len)
        if len(potential_name) > 2 and potential_name.lower() not in ['japan', 'usa', 'team']:
            player_names.append(potential_name)
    return player_names

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
        for user_id in matched_discord_ids: report_lines.append(f"- <@{user_id}>")
    else:
        report_lines.append("- (No attendees from the leaderboard were found in the database)")
    return "\n".join(report_lines)

def set_watermark(file_path):
    """
    Reads a logo file, encodes it to base64, and sets it as a full-page watermark.
    """
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
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Warning: Watermark file '{os.path.basename(file_path)}' not found. Please upload it to your GitHub repository.")

# --- Page Configuration and Theme/Watermark Application ---
st.set_page_config(
    page_title="Blitzmarine Event-Logger",
    page_icon=LOGO_FILE,
    layout="wide",
    initial_sidebar_state="collapsed",
)

set_watermark(LOGO_FILE)

# --- Streamlit GUI ---
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

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.discord_output = ""
    st.session_state.unique_names = []
    st.session_state.matched_ids = []

uploaded_files = st.file_uploader(
    "Upload one or more leaderboard screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.expander(f"View the {len(uploaded_files)} uploaded screenshot(s)..."):
        cols = st.columns(min(len(uploaded_files), 8))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 8]:
                st.image(uploaded_file, caption=f"Image {i+1}", width=150)
    
    st.write("---")

    if st.button("Generate Discord Report from All Screenshots"):
        if player_db_df.empty:
            st.warning("Cannot process because the player database file is missing, empty, or has incorrect columns.")
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
                st.session_state.unique_names = list(dict.fromkeys(all_extracted_names))
                with st.spinner("Step 2/3: Finding unique players and matching with database..."):
                    matched_ids = []
                    for name in st.session_state.unique_names:
                        discord_id = get_player_id_from_csv(name, player_db_df)
                        if discord_id:
                            matched_ids.append(str(discord_id))
                    st.session_state.matched_ids = list(dict.fromkeys(matched_ids))

                    with st.spinner("Step 3/3: Building final report..."):
                        st.session_state.discord_output = format_discord_report(st.session_state.matched_ids)
                        st.session_state.report_generated = True

if st.session_state.report_generated:
    st.write("---")
    st.subheader("✅ Your Combined Discord Report is Ready!")
    st.info("Click the copy icon in the top-right of the box below, then paste directly into Discord.")
    st.code(st.session_state.discord_output, language='markdown')

    with st.expander("See processing details"):
        st.write("**Unique Player Names Found (Combined & Deduplicated):**")
        st.dataframe(pd.DataFrame(st.session_state.unique_names, columns=["Identified Roblox Usernames"]))
        st.write("**Final Matched Discord User IDs:**")
        if st.session_state.matched_ids:
            st.dataframe(pd.DataFrame(st.session_state.matched_ids, columns=["Found Discord User IDs"]))
        else:
            st.warning("No players from any screenshot were found in the database.")

    if st.button("Start Over / Retry"):
        st.session_state.report_generated = False
        st.session_state.discord_output = ""
        st.session_state.unique_names = []
        st.session_state.matched_ids = []
        st.rerun()```
