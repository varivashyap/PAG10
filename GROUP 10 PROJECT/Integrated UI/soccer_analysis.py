import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
import sqlite3
import time
from ByteTrack import yolo_tracking
import cv2
from dataclasses import dataclass
from pathlib import Path
from action_detection import output_detection
 
# Retry parameters for the API
retry_attempts = 5
delay_seconds = 30
 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAW_Xc0Ds-EGE6iQfi_eW6p1auWdxwIv1g"
if not os.getenv("GOOGLE_API_KEY"):
    st.error("API key not found. Please set the `GOOGLE_API_KEY` environment variable.")
 
# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Function to generate analysis for a player using Gemini
def get_performance_analysis(player_data):
    prompt = (
        f"Analyze the performance of a soccer player with the following metrics:\n"
        f"Possession Percentage: {player_data['Possession Percentage']}%, "
        f"Goals Made: {player_data['Goals Made']}, "
        f"Fouls: {player_data['Number of Fouls']}, "
        f"Shots Off Target: {player_data['Shots Off Target']}, "
        f"Passes: {player_data['Number of Passes']}, "
        f"Team: {player_data['Team']}, "
        f"Interceptions: {player_data['Number of Intercepts']}.\n"
        "Provide a performance description, on the basis of the performance metrics give the position on which player performs and suggest improvements if needed."
        "Give only the description of the performance of the player on the basis of the metrics provided."
        "do not provide informations like analysis cannot be done with these metrics"
    )
    try:
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating analysis: {e}"
 
# Function to analyze team data (players)
def analyze_team(df, selected_player_id):
    # Clean column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')  # Drop empty rows
 
    # Map actual CSV column names to expected column names
    column_mapping = {
        "Player ID": "Player ID",
        "Possession": "Possession Percentage",
        "Goals Made": "Goals Made",
        "Fouls": "Number of Fouls",
        "Shots Off Target": "Shots Off Target",
        "Number of Passes": "Number of Passes",
        "Team": "Team",
        "Interceptions": "Number of Intercepts"
    }
    df.rename(columns=column_mapping, inplace=True)
 
    required_columns = [
        "Player ID",
        "Possession Percentage",
        "Goals Made",
        "Number of Fouls",
        "Shots Off Target",
        "Number of Passes",
        "Team",
        "Number of Intercepts"
    ]
 
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing in the uploaded CSV: {', '.join(missing_columns)}")
        return pd.DataFrame()  # Return empty DataFrame if columns are missing
 
    numeric_columns = [
        "Player ID",
        "Possession Percentage",
        "Goals Made",
        "Number of Fouls",
        "Shots Off Target",
        "Number of Passes",
        "Team",
        "Number of Intercepts"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
 
    player_data = df[df['Player ID'] == selected_player_id]
    if player_data.empty:
        st.error(f"No data found for Player ID: {selected_player_id}")
        return pd.DataFrame()  # Return empty DataFrame if player not found
    player_data = player_data.iloc[0]
 
    performance_data = {
        "Player ID": player_data["Player ID"],
        "Possession Percentage": player_data["Possession Percentage"],
        "Goals Made": player_data["Goals Made"],
        "Number of Fouls": player_data["Number of Fouls"],
        "Shots Off Target": player_data["Shots Off Target"],
        "Number of Passes": player_data["Number of Passes"],
        "Team": player_data["Team"],
        "Number of Intercepts": player_data["Number of Intercepts"]
    }
 
    # Get and display analysis
    analysis_text = get_performance_analysis(performance_data)
    st.markdown(f"**Performance Analysis for Player ID: {performance_data['Player ID']}**")
    st.write(analysis_text)
 
    return pd.DataFrame([{
        "Player ID": performance_data["Player ID"],
        "Team": performance_data["Team"],
        "Analysis": analysis_text
    }])
 
# Function to analyze team performance
def analyze_team_performance(df):
    # analyse the performance of the whole team
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')
 
    # store the analysis of each player from the get_performance_analysis function and store it in the csv file and also store it in the database
    # from that csv file analyse whole
 
    column_mapping = {
        "Player ID": "Player ID",
        "Possession": "Possession Percentage",
        "Goals Made": "Goals Made",
        "Fouls": "Number of Fouls",
        "Shots Off Target": "Shots Off Target",
        "Number of Passes": "Number of Passes",
        "Team": "Team",
        "Interceptions": "Number of Intercepts"
    }
 
    df.rename(columns=column_mapping, inplace=True)
 
    required_columns = [
        "Player ID",
        "Possession Percentage",
        "Goals Made",
        "Number of Fouls",
        "Shots Off Target",
        "Number of Passes",
        "Team",
        "Number of Intercepts"
    ]
 
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing in the uploaded CSV: {', '.join(missing_columns)}")
        return
    
    numeric_columns = [
        "Player ID",
        "Possession Percentage",
        "Goals Made",
        "Number of Fouls",
        "Shots Off Target",
        "Number of Passes",
        "Team",
        "Number of Intercepts"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
 
    # store the analysis of each player in the csv file
    analysis_data = []
    for index, row in df.iterrows():
        player_data = {
            "Player ID": row["Player ID"],
            "Possession Percentage": row["Possession Percentage"],
            "Goals Made": row["Goals Made"],
            "Number of Fouls": row["Number of Fouls"],
            "Shots Off Target": row["Shots Off Target"],
            "Number of Passes": row["Number of Passes"],
            "Number of Intercepts": row["Number of Intercepts"],
            "Team": row["Team"]
            
        }
        analysis_text = get_performance_analysis(player_data)
        analysis_data.append({
            "Player ID": row["Player ID"],
            "Team": row["Team"],
            "Analysis": analysis_text
        })
 
    analysis_df = pd.DataFrame(analysis_data)
    
    # Calculate team statistics
    total_goals = df["Goals Made"].sum()
    total_fouls = df["Number of Fouls"].sum()
    avg_possession_percentage = df["Possession Percentage"].mean()
    
    # Add team statistics to the analysis DataFrame
    analysis_df["Total Goals by Team"] = total_goals
    analysis_df["Total Fouls by Team"] = total_fouls
    analysis_df["Average Possession Percentage"] = avg_possession_percentage
    
    # take the analysis of each player from the above dataframe and generte the analysis of the whole team
    prompt = (
        f"Analyze the performance of a soccer player with the following metrics:\n"
        f"Player ID: {analysis_df['Player ID']}, "
        f"The team: {analysis_df['Team']}, "
        f"Analysis: {analysis_df['Analysis']}.\n"
        f"Total Goals by Team: {total_goals}, "
        f"Total Fouls by Team: {total_fouls}, "
        f"Average Possession Percentage: {avg_possession_percentage}%.\n"
        "Compare the performance of the whole team in both matches and identify the best-performing team."
        "compare on the base of analysis of each player."
        "analysis of team comparision should be done only on the basis of fouls and the analyis which was extacted from analysis_df"
        "provide whatever anaylis you can provide on the base of the analysis of each player and the metrics only."
    )
    try:
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(history=[])
        response = chat_session.send_message(prompt)
        st.write("**Team Performance Analysis**")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error in generating analysis: {e}")
 
    # store the analysis of each player in the database
    for index, row in analysis_df.iterrows():
        store_user_data(row["Player ID"], row["Analysis"])
 
    # store the analysis of the whole team in the csv file
    store_csv_file("team_analysis.csv", analysis_df.to_csv(index=False))
 
    st.download_button(
    label="Download Team Performance Analysis CSV",
    data=analysis_df.to_csv(index=False),
    file_name="team_performance_analysis.csv",
    mime="text/csv",
    key="team_performance_download"
)
 
    return analysis_df
 
 
# Function to generate match statistics analysis using Gemini
# Function to generate match statistics analysis using Gemini
def get_match_statistics(match_stats1_df, match_stats2_df):
    """
    This function generates analysis for the match statistics of two matches.
    1. Display match statistics for each match.
    2. Identify unique players in each match.
    3. Find common and unique players between the two matches.
    4. Compare the whole team in both matches to identify the best-performing team.
    5. Compare the performance of common players in both matches.
    6. Display summary insights.
    """
    
    # call the function to analyze the performance of the whole team in the two matches
    analyze_team_performance(match_stats1_df)
    analyze_team_performance(match_stats2_df)
    
    prompt = (
        "based on the analysis of the performance of the whole team in the two matches, from the function analyze_team_performance, give the analysis about how well a team performed in the different matches."
    )
 
    try:
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(history=[])
        response = chat_session.send_message(prompt)
        st.write("**Match Statistics Analysis**")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error in generating analysis: {e}")
 
    return response.text
 

# Function to display match statistics
def display_match_statistics(match_stats1_df, match_stats2_df):
    """
    This function displays match statistics for two DataFrames, representing
    two matches for a team. It compares the performance of different player sets
    to evaluate which combinations perform well together.
    """
    st.write("### Match Statistics and Team Performance Analysis")
 
 
    # Identifying unique players in each match
    match1_players = set(match_stats1_df['Player ID'])
    match2_players = set(match_stats2_df['Player ID'])
    
    # Find common and unique players between the two matches
    common_players = match1_players.intersection(match2_players)
    unique_to_match1 = match1_players - match2_players
    unique_to_match2 = match2_players - match1_players
    
    st.write("### Player Set Analysis")
    st.write(f"**Common Players in Both Matches:** {', '.join(map(str, common_players))}")
    st.write(f"**Unique Players in Match 1:** {', '.join(map(str, unique_to_match1))}")
    st.write(f"**Unique Players in Match 2:** {', '.join(map(str, unique_to_match2))}")
 
    # Example metric comparison: Goals Made and Possession Percentage
    st.write("### Performance Comparison of Common Players")
    if common_players:
        common_stats = []
        for player in common_players:
            match1_stats = match_stats1_df[match_stats1_df['Player ID'] == player]
            match2_stats = match_stats2_df[match_stats2_df['Player ID'] == player]
            common_stats.append({
                "Player ID": player,
                "Match 1 Goals": match1_stats["Goals Made"].values[0],
                "Match 2 Goals": match2_stats["Goals Made"].values[0],
                "Match 1 Possession %": match1_stats["Possession"].values[0],
                "Match 2 Possession %": match2_stats["Possession"].values[0]
            })
 
        common_stats_df = pd.DataFrame(common_stats)
        st.write("**Performance Comparison of Common Players Across Matches**")
        st.dataframe(common_stats_df)
 
    # Summary Insights
    st.write("### Summary Insights")
    st.write("Identify which set of players or specific player roles contribute to team success based on goals scored, possession, and other metrics.")
    match_stats_df = pd.concat([match_stats1_df, match_stats2_df], ignore_index=True)
    st.write("**Match Statistics**")
    st.dataframe(match_stats_df)
 
    # analyze the whole team performance and players performance in the two matches
    prompt = (
        f"Analyze the performance of a soccer player with the following metrics:\n"
        f"Possession Percentage: {match_stats_df['Possession']}%, "
        f"Goals Made: {match_stats_df['Goals Made']}, "
        f"Fouls: {match_stats_df['Number of Fouls']}, "
        f"Shots Off Target: {match_stats_df['Shots Off Target']}, "
        f"Passes: {match_stats_df['Number of Passes']}, "
        f"Interceptions: {match_stats_df['Number of Intercepts']}, "  
        f"Team: {match_stats_df['Team']}.\n"
        "Compare the performance of the whole team in both matches and identify the best-performing team."
    )
    try:
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(history=[])
        response = chat_session.send_message(prompt)
 
        st.write("**Team Performance Analysis**")
        st.write(response.text)
 
        return response.text
        
    except Exception as e:
        return f"Error in generating analysis: {e}"
    
def video_analysis_from_yolo(video_path, output_video_path):
    
    video = yolo_tracking.get_output_video(video_path,output_video_path)

    return video


 
    # Streamlit UI
def main():
        st.title("⚽ Soccer Player & Goalkeeper Performance Analysis")
 
        # Custom CSS for styling
        st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: transparent;
            }
            .stRadio > label {
                font-size: 18px;
                font-weight: bold;
            }
            .stRadio > div > label > div {
                background-color: transparent;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 5px;
            }
            .stRadio > div > label > div:hover {
                font-size: 20px;  /* Increase font size on hover */
            }
            </style>
        """, unsafe_allow_html=True)
 
        # Sidebar with tabs
        tab = st.sidebar.radio("Navigation", ["Home", "Player Performance", "Team Performance", "Match Statistics" , "Video Analysis"])
        if tab == "Home":
            st.header("Home")
            st.write("Welcome to the Soccer Player & Goalkeeper Performance Analysis tool. Use the sidebar to navigate through different sections.")
           
        elif tab == "Player Performance":
            st.header("Player Performance")
            # Upload player data
            team1_file = st.file_uploader("Upload Team 1 Player Data", type=["csv"])
            if team1_file:
                team1 = pd.read_csv(team1_file, skip_blank_lines=True)
                team1.columns = team1.columns.str.strip()
                team1 = team1.dropna(how='all')
 
                column_mapping = {
                    "Player ID": "Player ID",
                    "Possession": "Possession Percentage",
                    "Goals Made": "Goals Made",
                    "Goals": "Goals Made",
                    "Fouls": "Number of Fouls",
                    "Shots Off Target": "Shots Off Target",
                    "Number of Passes": "Number of Passes",
                    "Interceptions": "Number of Intercepts"
                }
                team1.rename(columns=column_mapping, inplace=True)
 
                required_columns = [
                    "Player ID",
                    "Possession Percentage",
                    "Goals Made",
                    "Number of Fouls",
                    "Shots Off Target",
                    "Number of Passes",
                    "Number of Intercepts"
                ]
                missing_columns = [col for col in required_columns if col not in team1.columns]
                if missing_columns:
                    st.error(f"The following required columns are missing in the uploaded CSV: {', '.join(missing_columns)}")
                    return
 
                numeric_columns = [
                    "Player ID",
                    "Possession Percentage",
                    "Goals Made",
                    "Number of Fouls",
                    "Shots Off Target",
                    "Number of Passes",
                    "Number of Intercepts"
                ]
                for col in numeric_columns:
                    team1[col] = pd.to_numeric(team1[col], errors='coerce')
 
                team1 = team1.dropna(subset=['Player ID'])
                team1['Player ID'] = team1['Player ID'].astype(int)
                player_ids = team1['Player ID'].unique().tolist()
                selected_player_id = st.selectbox("Select a Player ID to Analyze", player_ids)
 
                if st.button("Analyze Player Performance"):
                    player_analysis = analyze_team(team1, selected_player_id)
                    if not player_analysis.empty:
                        st.download_button(
                                            label="Download Performance Analysis CSV",
                                            data=player_analysis.to_csv(index=False),
                                            file_name="performance_analysis.csv",
                                            mime="text/csv",
                                            key="player_performance_download"
)
 
 
        elif tab == "Team Performance":
            st.header("Team Performance")
            # Upload team data
            team_file = st.file_uploader("Upload Team Data", type=["csv"])
            if team_file:
                team_df = pd.read_csv(team_file, skip_blank_lines=True)
                analyze_team_performance(team_df)

 
        elif tab == "Match Statistics":
            # match statistics will give analysis of the performance of the team in different matches a single team can have different players for different matches so goal is to whic set of players perform well when they play together
            st.header("Match Statistics")
            st.write("Upload match statistics for two matches to compare the performance of the team.")
 
            # Upload match statistics for Match 1
            match_stats1_file = st.file_uploader("Upload Match Statistics for Match 1", type=["csv"])
            match_stats2_file = st.file_uploader("Upload Match Statistics for Match 2", type=["csv"])
            match_stats3_file = st.file_uploader("Upload Match Statistics for Match 3", type=["csv"])
 
            if match_stats1_file and match_stats2_file and match_stats3_file:
                match_stats1_df = pd.read_csv(match_stats1_file)
                match_stats2_df = pd.read_csv(match_stats2_file)
                match_stats3_df = pd.read_csv(match_stats3_file)
                
                display_match_statistics(match_stats1_df, match_stats2_df)
                get_match_statistics(match_stats1_df, match_stats2_df)
                
                display_match_statistics(match_stats2_df, match_stats3_df)
                get_match_statistics(match_stats2_df, match_stats3_df)
 
                display_match_statistics(match_stats1_df, match_stats3_df)
                get_match_statistics(match_stats1_df, match_stats3_df)
           
        
       
        elif tab == "Video Analysis":
            st.header("Video Analysis")
            st.write("Upload a video file or provide a video URL for analysis.")
 
            # Option to upload video file
            video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
            def save_video(video_file):
                # create a directory to save the video file
                if not os.path.exists("/home/urvashi2022/Desktop/UI_DEVELOPMENT/API/input"):
                    os.makedirs("/home/urvashi2022/Desktop/UI_DEVELOPMENT/API/input")
                if not os.path.exists("/home/urvashi2022/Desktop/UI_DEVELOPMENT/API/output"):
                    os.makedirs("/home/urvashi2022/Desktop/UI_DEVELOPMENT/API/output")

                video_path = os.path.join("/home/urvashi2022/Desktop/UI_DEVELOPMENT/API/input", video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                return video_path
            if video_file is not None:
                st.write("**Input Video:**")
                video_path = save_video(video_file)
                # output_video_path = Path("/home/urvashi2022/Desktop/UI_DEVELOPMENT/tracking/output1.mp4")

                # video_analysis_from_yolo(video_path, output_video_path)
                # st.write("**Output Video:**")

                # # Display the output video
                # st.video(str(output_video_path))

                st.write("Action Detection")
                df = output_detection(video_path)
                st.write("Download the detection results:")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='detection_results.csv',
                    mime='text/csv',
    )


 
            # Option to provide video URL
            video_url = st.text_input("Or enter Video URL")
            if video_url:
                st.write("**Input Video:**")
                st.video(video_url)

            
if __name__ == "__main__":
    def init_db():
        conn = sqlite3.connect('soccer_analysis.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                analysis TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS csv_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                file_content BLOB
            )
        ''')
        conn.commit()
        conn.close()
 
    def store_user_data(player_id, analysis):
        conn = sqlite3.connect('soccer_analysis.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_data (player_id, analysis)
            VALUES (?, ?)
        ''', (player_id, analysis))
        conn.commit()
        conn.close()
 
    def store_csv_file(file_name, file_content):
        conn = sqlite3.connect('soccer_analysis.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO csv_files (file_name, file_content)
            VALUES (?, ?)
        ''', (file_name, file_content))
        conn.commit()
        conn.close()
 
    init_db()
    main()