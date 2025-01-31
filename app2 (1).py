import streamlit as st
import openpyxl
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import os
import threading

# Configure Gemini API
api_key = "AIzaSyDcaMWRY-8VMU0wg5HOl0ZUP6aDYvIaSkk"
genai.configure(api_key=api_key)

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Function to create or load the CRM Excel file
def create_or_load_excel():
    if os.path.exists("CC.xlsx"):
        workbook = openpyxl.load_workbook("CC.xlsx")
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append([
            "Name", "Email", "Phone", "Company Name", "Deal ID", "Date of Interaction",
            "User Complaint", "Recommendations", "Deal Recommendations", "Post-Call Summary"
        ])
    return workbook

# Function to save data to the Excel file
def save_to_excel(user_details, user_complaint, recommendations, deal_recommendations, postcall_summary):
    try:
        workbook = create_or_load_excel()
        sheet = workbook.active

        sentiment_score = analyzer.polarity_scores(user_complaint)["compound"]
        sentiment = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

        deal_id = f"DEAL-{int(datetime.now().timestamp())}"
        date_of_interaction = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sheet.append([
            user_details['name'], user_details['email'], user_details['phone'],
            user_details['company'], deal_id, date_of_interaction,
            user_complaint, recommendations, deal_recommendations, postcall_summary
        ])

        workbook.save("CC.xlsx")
        return "Data saved successfully."
    except Exception as e:
        return f"Error saving data: {str(e)}"

# Function to analyze user input using Gemini API
def analyze_audio(text_input):
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=(
                "You are the recommendation system to do the works like: "
                "You need to analyze the voice input of the user and give the output like: "
                "sentiment: analysis by the user input, intent: analysis by the user input, "
                "tone: analysis by the user input, "
                "recommendation: analysis by the user input, "
                "deal recommendation: analysis by the user input and give deal recommendations of 5 points, "
                "recommendations: analysis by the user input and give recommendations of 10 points, "
                "postcall analysis: analysis of the conversation and give it in only 3 lines."
            ),
        )

        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [text_input]}]
        )

        response = chat_session.send_message(text_input)
        return response.text
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

# Function to speak text using pyttsx3
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to record audio using microphone
def record_audio(prompt):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Display and speak the prompt
    st.write(f"**{prompt}**")
    threading.Thread(target=speak_text, args=(prompt,)).start()

    listening_placeholder = st.empty()  # Placeholder for dynamic updates

    with microphone as source:
        listening_placeholder.info("Listening for your input...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        try:
            audio = recognizer.listen(source)
            listening_placeholder.success("Processing your voice input...")
            text = recognizer.recognize_google(audio)
            st.write(f"**You said:** {text}")
            listening_placeholder.empty()  # Clear the placeholder
            return text
        except sr.UnknownValueError:
            listening_placeholder.error("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            listening_placeholder.error("Sorry, there was an issue with the speech service.")
            return None

# Streamlit UI
st.title("Real-Time AI Sales Intelligence and Dynamic Deal Recommendation System")

if st.button("Start Conversation"):
    with st.spinner("Initiating conversation..."):
        user_details = {}

        # Ask for name
        user_details['name'] = record_audio("What is your name?")
        if not user_details['name']:
            st.error("Failed to capture your name.")
            st.stop()

        # Ask for email
        user_details['email'] = record_audio("What is your email address?")
        if not user_details['email']:
            st.error("Failed to capture your email.")
            st.stop()

        # Ask for phone number
        user_details['phone'] = record_audio("What is your phone number?")
        if not user_details['phone']:
            st.error("Failed to capture your phone number.")
            st.stop()

        # Ask for company name
        user_details['company'] = record_audio("What is your company name?")
        if not user_details['company']:
            st.error("Failed to capture your company name.")
            st.stop()

        # Ask for complaint
        complaint = record_audio("What is your complaint?")
        if not complaint:
            st.error("Failed to capture your complaint.")
            st.stop()

        # Analyze the complaint
        recommendations = analyze_audio(complaint)
        deal_recommendations = "Offer personalized discount on the next purchase."
        postcall_summary = "The user had issues with product reliability. Recommended offering a discount on future purchases."

        # Save to Excel
        save_status = save_to_excel(user_details, complaint, recommendations, deal_recommendations, postcall_summary)

        st.success("Conversation completed!")
        st.subheader("Generated Results")
        st.write("**Recommendations:**", recommendations)
        st.write("**Deal Recommendations:**", deal_recommendations)
        st.write("**Post-Call Summary:**", postcall_summary)
        st.success(save_status)
