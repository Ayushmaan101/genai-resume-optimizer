import os
from dotenv import load_dotenv

print("--- Starting Environment Debugger ---")

# 1. Print the current directory to confirm where the script is running
try:
    cwd = os.getcwd()
    print(f"\n✅ 1. Current Working Directory: {cwd}")
except Exception as e:
    print(f"\n❌ ERROR getting current directory: {e}")

# 2. Check if the .env file exists in this directory
env_file_path = os.path.join(cwd, '.env')
file_exists = os.path.exists(env_file_path)
print(f"\n✅ 2. Does '.env' file exist here? -> {file_exists}")

if not file_exists:
    print("\n   [DIAGNOSIS] The '.env' file was not found. Please ensure it's in the same folder as this script and named correctly.")
else:
    # 3. Read and print the raw content to check for formatting errors
    print("\n✅ 3. Reading raw content from '.env' file:")
    try:
        with open(env_file_path, 'r') as f:
            content = f.read()
            print("--- FILE CONTENT ---")
            print(content)
            print("--- END OF FILE CONTENT ---\n")
        
        if "GOOGLE_API_KEY" not in content:
             print("   [DIAGNOSIS] The text 'GOOGLE_API_KEY' was not found in the file. Check for typos.\n")

    except Exception as e:
        print(f"❌ ERROR: Could not read the .env file. Error: {e}\n")

# 4. Try to load the .env file and get the variable
print("✅ 4. Attempting to load the key using python-dotenv...")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"\n✅ 5. Value retrieved for GOOGLE_API_KEY: '{api_key}'")

if not api_key:
    print("\n   [FINAL DIAGNOSIS] The key is NOT loading. This is likely a formatting error in the .env file or a permissions issue.")
    print('   The content MUST be exactly: GOOGLE_API_KEY="YOUR_KEY_HERE"')
else:
    print("\n   [FINAL DIAGNOSIS] SUCCESS! The key was loaded correctly by this script.")

print("\n--- Debugger Finished ---")