# Player Analysis Model
This project focuses on analyzing players' performance by leveraging metrics such as goals, fouls, possession, etc., stored in CSV files. The analysis is powered by the Gemini 1.5 Flash model, which processes the data and provides comprehensive insights.

To utilize this model, an API call is made from the local system. Users need to generate their own API key (free of cost) and insert it into the appropriate file to run the application.

Setup and Usage Instructions
## Step 1: Prerequisites
Download all the project folders.
Install the required dependencies for each folder, which are listed in the respective requirements.txt files.
## Step 2: Integration
Move all folders into the Integrated UI directory.
Locate the file named soccer_analysis.py in the Integrated UI directory.
## Step 3: API Key Setup
Generate your free API key from the service provider (Gemini 1.5 Flash).
Open the soccer_analysis.py file and replace the placeholder for the API key with your own key.
## Step 4: Running the Application
Open the terminal and navigate to the Integrated UI folder.
Run the following command:
bash
streamlit run soccer_analysis.py
The application will launch in your default web browser.
