import os
import torch

# ------------------------
# Configuration
# ------------------------
PDF_DIR = r"C:\Users\Alina.Javed\Documents\PPA AI\GenAI_PPA\1-Alina_Working\Alina_Working\CPPA PPA PDF"
CHROMA_DIR = r"C:\Chroma"
ABBREV_FILE = os.path.join(CHROMA_DIR, "abbreviations.json")

# HuggingFace embeddings
EMBEDDINGS_MODEL = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Google API Key
GOOGLE_API_KEY = "AIzaSyAQ67pr0uC5nwlfuJorkIonbmW0QgIclWU"

 