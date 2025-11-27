\# Brain Tumor Detection System with Encryption



\*\*Short description:\*\*  

A brain tumor detection system using handcrafted MRI feature extraction and an SVM classifier, integrated with a Flask web app for secure encrypted uploads, decryption, prediction, and visualization.



---



\## Features

\- MRI preprocessing and handcrafted feature extraction

\- SVM-based tumor classification and severity analysis

\- Chaos-based image encryption \& decryption module

\- Flask web interface for secure upload, prediction, and result rendering

\- Requirements \& reproducible environment instructions



---



\## Tech stack

\- Python (3.8+)

\- scikit-learn (SVM)

\- OpenCV / Pillow (image processing)

\- Flask (web UI)

\- joblib (model serialization)

\- Chaos-based encryption module (custom)



---



\## Setup (Local)



\### 1. Clone the repo

```bash

git clone https://github.com/clmesantu/BrainTumorDetectionSystem.git

cd BrainTumorDetectionSystem



\### 2. Create \& activate a virtual environment



python -m venv venv

\# Windows (PowerShell)

.\\venv\\Scripts\\Activate.ps1

\# macOS / Linux

source venv/bin/activate



\### 3. Install dependencies



pip install -r requirements.txt



---



\## ⚠️ Train Model (Required Before Running App)



> The SVM model \*\*must be trained first\*\* using `train\_and\_save\_model.py`.  

> Once trained, `app.py` will load the generated `svm\_model.joblib` file from the `model/` folder.



\### Train the Model

```bash

\# ensure venv is activated

python scripts/train\_and\_save\_model.py 

output model/svm\_model.joblib

```



\### Then Run the App

```bash

python app.py

```



> If the model file is missing, the Flask app will not run.  

> Make sure `model/svm\_model.joblib` exists after training.



---



\## Usage



\### Run Flask App

```bash

python app.py

```

Open: \*\*http://127.0.0.1:5000/\*\*  

Use the interface to upload encrypted or raw MRI images.



---



\## Encrypt / Decrypt (CLI)

```bash

python scripts/encrypt.py --input input.png --key KEY --output encrypted.bin

python scripts/decrypt.py --input encrypted.bin --key KEY --output output.png

```



---


## Contact

Santhosh S  

Email: \*\*santhoshs382004@gmail.com\*\*  

LinkedIn: \*\*https://www.linkedin.com/in/santhoshs-538707224/\*\*  

GitHub: \*\*https://github.com/clmesantu\*\*

