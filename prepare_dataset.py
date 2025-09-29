import os
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader

DATASET_DIR = r"C:\Users\mouni\OneDrive\Desktop\project 1\P-344 Dataset\Resumes_Docx"

texts = []
labels = []

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            text = ""
            try:
                if file.lower().endswith(".docx"):
                    # Only try to open valid .docx files
                    doc = Document(file_path)
                    text = "\n".join(p.text for p in doc.paragraphs)

                elif file.lower().endswith(".pdf"):
                    reader = PdfReader(file_path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)

                else:
                    print(f"⚠ Skipping unsupported file type: {file}")
                    continue  # skip non-docx/pdf files

            except Exception as e:
                print(f"❌ Error reading file {file}: {e}")
                continue  # skip files that cause errors

            if text.strip():  # only add if text is not empty
                texts.append(text)
                labels.append(folder)

# Save to CSV
df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("dataset.csv", index=False, encoding="utf-8")

print("✅ dataset.csv created successfully with", len(df), "records")
