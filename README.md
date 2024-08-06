# WARNING
Due to the limited time and to maximize programming efficiency, all the backend code needed to run the PII Redactor have been saved to a single file. Since the hackathon was very **build fast & break fast**, no vENVs were used & best practices were generally ignored. To make the code readable, we have added comments wherever required. The code has also been linted. A simple uvicorn server created with FastAPI exposes the API. However, the server is in development mode & has (0.0.0.0) CORS access. 

# PREREQUISITIES
```
pip install -r requirements.txt
```
This will downlaod all non-std modules required for the program to run. GhostScript, ImageMagick, and Tesseract must also be installed as executables. A trained Haar Classifier XML is available in this repository, which is required for face-blur. Once done, the server can be spun up with - 
```
cd path/to/folder && python main.py
```
