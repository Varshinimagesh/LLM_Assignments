#o/p ok crm saving 
import os
import time
import pyaudio
import speech_recognition as sr
import google.generativeai as genai
import openpyxl
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure Gemini API
api_key = "AIzaSyDcaMWRY-8VMU0wg5HOl0ZUP6aDYvIaSkk"
genai.configure(api_key=api_key)

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Create or load the Excel file
def create_or_load_excel():
    """Creates or loads the CRM Excel file."""
    if os.path.exists("Crm_data.xlsx"):
        workbook = openpyxl.load_workbook("Crm_data.xlsx")
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append([
            "Name", "Email", "Phone", "Company Name", "Deal ID", "Date of Interaction",
            "Sentiment Score", "User Complaint", "Recommendations", "Deal Recommendations", "Post-Call Summary"
        ])
    return workbook

def record_audio(prompt=None):
    """Capture live audio from the microphone and return it as text."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    if prompt:
        print(prompt)
    
    print("Adjusting for ambient noise... Please wait.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=5)  # Adjust ambient noise level

    print("Listening for your command...")
    with microphone as source:
        try:
            audio = recognizer.listen(source)
            print("Processing your voice input...")
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue with the speech service.")
            return None

def analyze_audio(text_input):
    """Send text input to Gemini API for analysis."""
    # Create the model configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        system_instruction=( 
            "You are the recommendation system to do the works like: "
            "You need to analyze the voice input of the user and give the output like: "
            "sentiment: analysis by the user input, intent: analysis by the user input, "
            "tone: analysis by the user input, "
            "recommendation: analysis by the user input, "
            "deal recommendation: analysis by the user input and give recommendations of 5 points, "
            "postcall analysis: analysis of the conversation and give it in only 2 lines."
        ),
    )

    # Start chat session
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [text_input]}]
    )

    # Send the text to the model and get the response
    response = chat_session.send_message(text_input)
    analysis_result = response.text
    print(f"Analysis Result: {analysis_result}")
    return analysis_result

def save_to_excel(user_details, user_complaint, recommendations, deal_recommendations, postcall_summary):
    """Save the results to the Excel file."""
    workbook = create_or_load_excel()
    sheet = workbook.active

    # Extract sentiment analysis using VADER
    sentiment_score = analyzer.polarity_scores(user_complaint)["compound"]
    sentiment = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
    
    # Generate Deal ID and Date of Interaction
    deal_id = f"DEAL-{int(datetime.now().timestamp())}"
    date_of_interaction = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add the new row to the sheet
    sheet.append([user_details['name'], user_details['email'], user_details['phone'],
                  user_details['company'], deal_id, date_of_interaction, sentiment_score,
                  user_complaint, recommendations, deal_recommendations, postcall_summary])

    # Save the workbook
    workbook.save("Crm_data.xlsx")
    print("Data saved to Crm_data.xlsx")

def main():
    """Main function to capture and process live voice input and user details."""
    user_details = {}

    print("\nPlease provide your details...")

    # Capture user name
    user_details['name'] = record_audio("Please say your name.")
    if not user_details['name']:
        return

    # Capture user email
    user_details['email'] = record_audio("Please say your email address.")
    if not user_details['email']:
        return

    # Capture user phone number
    user_details['phone'] = record_audio("Please say your phone number.")
    if not user_details['phone']:
        return

    # Capture user company name
    user_details['company'] = record_audio("Please say your company name.")
    if not user_details['company']:
        return

    print("\nNow, How may I help you ?")
    user_complaint = record_audio()
    if not user_complaint:
        return

    print("\nNow, I will generate recommendations based on your complaint.")
    recommendations = analyze_audio(user_complaint)
    
    # Deal Recommendations (Simulating for now as an example)
    deal_recommendations = "Offer personalized discount on the next purchase."

    # Post-call Summary (Simulating for now as an example)
    postcall_summary = "The user had issues with product reliability. Recommended offering a discount on future purchases."

    # Save the results to the Excel file
    save_to_excel(user_details, user_complaint, recommendations, deal_recommendations, postcall_summary)
    
    print("Execution completed. Exiting program.")

if __name__ == "__main__":
    main()
