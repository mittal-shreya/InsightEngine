# InsightEngine  
A **Streamlit-based application** that automates survey response validation using **rule-based filters** and **AI models** like **RoBERTa** and **DistilBERT**.  


## Features  

### **File Upload & Data Ingestion**  
 * Uploads an **Excel file** with survey responses.  
 * Reads the file and **standardizes column names** for consistency.  

### **Preprocessing & Text Cleaning**  
* Converts text to **lowercase** for uniform processing.  
* **Removes special characters** and unwanted terms to clean responses.  

### **Rule-Based Filtering (Tier 1)**  
* **Detects Gibberish:** Identifies random characters and excessive repetition.  
* **Identifies Profanity:** Uses a **custom profanity filter**, excluding liquor-related words.  

### **Contextual Analysis (Tier 2)**  
* **Relevance Check:** Uses **RoBERTa (Zero-Shot Classification)** to verify if responses match the question.  

### **Deep Learning & AI Detection (Tier 3)**  
* **DistilBERT Model:** Predicts whether a response is **low-quality**.  
* **RoBERTa Model:** Assigns **relevance scores** to each answer.  

### **Final Flagging & Report Generation**  
* Flags responses based on **rule-based & AI checks**.  
* **Streamlit Dashboard** displays flagged vs. clean responses.  
* **Downloadable Reports** (Full dataset or flagged entries).  

---

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/market-survey-analysis.git
cd market-survey-analysis
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:

bash
```
streamlit run app.py
```
