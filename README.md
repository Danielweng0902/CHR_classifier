# 📘 CHR_Classifier

[中文版本](README_zh.md)

## 📖 Overview
CHR Classifier is an OCR-based pipeline designed for recognizing and extracting **Traditional Chinese handwriting** from scanned cram school worksheets.  
It processes scanned images, detects grid structures, classifies characters with OCR and whitelist inference, and saves cropped handwriting samples into datasets.  

---

## 🏫 Project Background
This project was originally developed for the **WASN Lab cram school program**, where scanned vocabulary practice books were provided.  
The system serves as a **data labeling and classification tool**, aiming to collect common Traditional Chinese characters for downstream research.  

⚠️**Note**: Due to project restrictions, the original cram school dataset is **not publicly available**.  
If you want to use this project, please prepare your **own scanned worksheets or documents** as input. Thank you for your understanding.  

- Current **grid detection & coverage rate**: **99.99%**  
- Current **OCR classification accuracy**: ~**95%**  
- OCR engine: **TesseractOCR** (fine-tuned for Traditional Chinese)

---

## ✨ Features
- 🧩 **Grid detection**: multi-channel approach (contours, Hough transform, projection profile).  
- 🔍 **Strict blank detection**: prevents saving empty/noisy cells via multi-feature QC (persistence mask, edge density, connected components).  
- 📝 **Whitelist inference**: supports character sequences with optional user-defined starting anchors.  
- 📊 **Detailed statistics report**: includes storage rate, data yield rate, and incomplete column logs.  
- ⚡ **Automation**: auto-runs preprocessing (`pdf2png.py`, `preprocess_pages.py`) if needed.  

---

## 🗂 File Structure

CHR_classifier

├── main.py # main code

├── config.py # all the config

├── whitelist.py # whitelist process

├── whitelist.txt # Character whitelist

├── pdf2png.py # Convert PDF to PNG if needed

├── preprocess_pages.py # Page preprocessing utility

├── detect_grid.py # Grid detection debugger

├── ocr.py # Optical Character Recognition and preprocess to Chinese words 

├── report.py # output statistics

├── data/ # folder originally put each page as .png file

└── pdf/ # folder originally put the pages as .pdf file

└── datasets/ to output the folders of result


---

## 🔧 Requirements

   ```bash
   pip install -r requirements.txt
   ```

- Python 3.8+
- numpy>=1.21.0
- [opencv-python>=4.5.0](https://opencv.org/)
- pdf2image>=1.16.3
- Pillow>=9.0.0
- bayesian-optimization
- [pytesseract>=0.3.10](https://github.com/madmaze/pytesseract)

  
  ⚠️ **Important**: Must have **Tesseract OCR installed locally** and set the correct path in your code, e.g.:  
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

---

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Daniel930902/CHR_classifier
   ```
2. Prepare scanned PDF file inside "pdf" folder or PNG pages inside "data/{pdf/pages name}/".

  ( ⚠️the original dataset is not publicly available. Prepare your **own scanned worksheets or documents** as input. )

3. Enter the project directory
   ```bash
   cd CHR_classifier
   ```
4. Run the classifier by the command below:
   ```bash
   python CHR_classifier.py
   ```
5. Results (cropped handwriting images and debug visualizations) will be saved to:

    ex.

    ```
    ./CHR_classifier/{output_folder}/

    ```

    
    ```
    ./CHR_classifier/250928/
    ```

---

## 🔄 Processing Flow


```mermaid

flowchart TD
    A[Scanned PDF/PNG Pages] --> B["Preprocessing: pdf2png & preprocess_pages"]
    B --> C["Grid Detection: Contours / Hough / Projection"]
    C --> D["Label Row OCR + Whitelist Inference"]
    D --> E["Dynamic Blank Check (multi-feature)"]
    E --> F["Save Cropped Handwriting Images"]
    F --> G["Statistics Report"]


```


---

## 📊 Example Output

* Cropped handwriting images organized by character

* Debug visualizations of label OCR & grid detection

* Final statistics report with storage rate and yield rate

---

🙏 Acknowledgment

This project originally uses the fine-tuned Traditional Chinese Tesseract model from:
[ tessdata_chi ]( gumblex/tessdata_chi )

---

📝 Note

This README was drafted and refined with assistance from ChatGPT 5.

