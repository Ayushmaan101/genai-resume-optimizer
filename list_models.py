import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables to get the API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("🚨 API Key not found. Please ensure your .env file is correct.")
else:
    try:
        print("Authenticating with Google...")
        genai.configure(api_key=api_key)
        print("Authentication successful.")

        print("\n--- Available Models for Content Generation ---")
        # List all models and filter for the ones that support 'generateContent'
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        print("---------------------------------------------")
        print("\nIf this list is empty, your project has no generative models enabled.")

    except Exception as e:
        print(f"An error occurred: {e}")