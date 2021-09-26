import os

# warn: please run python app.py in terminal first
os.system("python ingestion.py")
os.system("python training.py")
os.system("python scoring.py")
os.system("python reporting.py")
os.system("python apicalls.py")