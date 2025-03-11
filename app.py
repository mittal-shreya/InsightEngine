import streamlit as st
import pandas as pd
import torch
import numpy as np
from better_profanity import profanity
import importlib.resources
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# -------------------------------------------------------------------------
# Set up environment and page configuration
# -------------------------------------------------------------------------

# Disable Weights & Biases logging to avoid unnecessary output
os.environ["WANDB_DISABLED"] = "true"

# Configure the Streamlit app's page title, layout, and display a logo and title
st.set_page_config(page_title="Market Survey Analysis", layout="wide")
st.image("logo.jpg", width=100)
st.title("Market Survey Analysis")

# -------------------------------------------------------------------------
# Custom Profanity Filter Setup
# -------------------------------------------------------------------------

def get_filtered_profanity_list():
    """
    Reads the default profanity wordlist from the better_profanity package and removes
    words related to alcoholic beverages. This allows those words to be exempt from filtering.
    
    Returns:
        A list of profanity words with the specified words removed.
    """
    # List of alcohol-related words to remove from the profanity list
    words_to_remove = [
        'drink', 'drank', 'drunk', 'booze', 'brew', 'liquor', 'alcohol', 
        'beverage', 'pub', 'bar', 'tavern', 'brewery', 'distillery',
        'cocktail', 'shot', 'chug', 'sip', 'hangover', 'intoxicated',
        'rum', 'vodka', 'whiskey', 'whisky', 'tequila', 'gin', 'beer', 'wine',
        'brandy', 'sake', 'schnapps', 'absinthe', 'moonshine', 'arrack',
        'feni', 'toddy', 'desi daru', 'country liquor', 'neutral spirit',
        'smirnoff', 'bacardi', 'captain', 'morgan', 'jack', 'daniels',
        'absolut', 'jagermeister', 'hennessy', 'chivas', 'ballantines',
        'black label', 'red label', 'black dog', 'teachers', 'glenfiddich',
        'officers choice', 'royal stag', 'imperial blue', 'antiquity',
        'mcdowells', 'bagpiper', 'haywards', 'old monk', 'hercules',
        'directors special', '8pm', 'original choice', 'signature',
        'blenders pride', 'royal challenge', 'kingfisher', 'tuborg',
        'bira', 'white mischief', 'rockford', 'magic moments',
        'lager', 'ale', 'stout', 'pilsner', 'ipa', 'draught', 'craft beer'
    ]
    
    # Open and read the default profanity wordlist from the package
    with importlib.resources.open_text("better_profanity", "profanity_wordlist.txt") as f:
        default_words = [word.strip().lower() for word in f.read().splitlines()]
    
    # Return a list excluding the specified alcohol-related words (ignoring case)
    return [w for w in default_words if w not in {word.lower() for word in words_to_remove}]

# Load the custom profanity list into the profanity module
profanity.load_censor_words(get_filtered_profanity_list())
# Create a set of words to check against for filtering out liquor-related words
excluded_liquor_words = set(get_filtered_profanity_list())

# -------------------------------------------------------------------------
# Rule-Based Detection Functions
# -------------------------------------------------------------------------

def is_gibberish(text):
    """
    Determines if the given text is gibberish based on several heuristics:
    - Short texts are assumed not to be gibberish.
    - If text contains any excluded liquor words, it's not considered gibberish.
    - A high proportion of non-alphabetical characters.
    - Excessive character repetitions.
    - Presence of common keyboard patterns (e.g., 'qwerty').
    - Very low proportion of vowels.
    
    Args:
        text (str): The text to evaluate.
    
    Returns:
        bool: True if text is likely gibberish, False otherwise.
    """
    if not isinstance(text, str) or len(text) < 5:
        return False
    # Split text into words and check if any are in the allowed liquor words
    text_words = text.lower().split()
    if any(word in excluded_liquor_words for word in text_words):
        return False
    # Check ratio of non-alphabet characters to overall length
    non_alpha = sum(1 for c in text if not c.isalpha())
    if non_alpha / len(text) > 0.5:
        return True
    # Check for excessive consecutive repeated characters
    repeats = sum(1 for i in range(len(text) - 1) if text[i] == text[i+1])
    if repeats / len(text) > 0.3:
        return True
    # Check for common keyboard patterns
    patterns = ['qwerty', 'asdf', 'zxcv', '12345', '1qaz', '2wsx']
    if any(p in text.lower() for p in patterns):
        return True
    # Check for very low vowel count
    vowels = 'aeiouAEIOU'
    vowel_count = sum(1 for c in text if c in vowels)
    if vowel_count / len(text) < 0.1:
        return True
    return False

def contains_profanity(text):
    """
    Uses the better_profanity package to determine if the text contains profanity.
    Returns False for empty or non-string inputs.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if profanity is found, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return profanity.contains_profanity(text)

# -------------------------------------------------------------------------
# Define Expected Columns and Mapping for Survey Questions
# -------------------------------------------------------------------------

# List of survey question columns to analyze
columns_to_check = [
    'Q16A. What is the most important thing you LIKE about the shown concept?',
    'Q16B. What is the most important thing you DISLIKE about the shown concept?',
    'Q18_1 What specific product that you are currently using would the shown product replace?',
    'Q18_2 What specific product that you are currently using would the shown concept replace?',
    'Q18_3 What specific product that you are currently using would the shown concept replace?'
]

# Mapping from partial column name patterns to the expected full column names
column_mapping = {
    'Q16A. What is the most important thing you LIKE': columns_to_check[0],
    'Q16B. What is the most important thing you DISLIKE': columns_to_check[1],
    'Q18_1 What specific product': columns_to_check[2],
    'Q18_2 What specific product': columns_to_check[3],
    'Q18_3 What specific product': columns_to_check[4]
}

# -------------------------------------------------------------------------
# Machine Learning (ML) Model Functions
# -------------------------------------------------------------------------

@st.cache_resource
def load_ml_models():
    """
    Loads and caches the ML models to avoid reloading them on each app rerun.
    Loads:
      - A RoBERTa model for zero-shot classification to determine relevance.
      - A DistilBERT model for quality prediction from a local directory ('final_model').
    
    Returns:
        tuple: (roberta_classifier, distilbert_tokenizer, distilbert_model)
    """
    # Display CUDA availability status in the sidebar
    cuda_status = "Available" if torch.cuda.is_available() else "Not Available"
    st.sidebar.write(f"CUDA Status: {cuda_status}")
    
    # Determine device: GPU if available, else CPU (-1 for pipeline)
    device = 0 if torch.cuda.is_available() else -1
    
    # Load RoBERTa model for zero-shot classification
    roberta_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=device)
    
    # Load DistilBERT model and tokenizer from the specified directory
    distilbert_path = "./final_model"
    try:
        distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_path)
        distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_path)
        # Move model to GPU if available for faster predictions
        if torch.cuda.is_available():
            distilbert_model = distilbert_model.to(device)
        st.sidebar.success("DistilBERT model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load DistilBERT model: {str(e)}")
        distilbert_tokenizer = None
        distilbert_model = None
    
    return roberta_classifier, distilbert_tokenizer, distilbert_model

def extract_question_text(q):
    """
    Strips the question identifier (e.g., 'Q16A.', 'Q18_1') from the full question text.
    
    Args:
        q (str): The full question string.
    
    Returns:
        str: The question text without the identifier.
    """
    if q.startswith('Q16A.'):
        return q[len('Q16A. '):]
    elif q.startswith('Q16B.'):
        return q[len('Q16B. '):]
    elif q.startswith('Q18_1'):
        return q[len('Q18_1 '):]
    elif q.startswith('Q18_2'):
        return q[len('Q18_2 '):]
    elif q.startswith('Q18_3'):
        return q[len('Q18_3 '):]
    return q

def get_relevance_score(classifier, question, answer):
    """
    Computes the relevance score of an answer given the question using a zero-shot classifier.
    The function handles very long inputs by truncating the text appropriately.
    
    Args:
        classifier: The RoBERTa zero-shot classification pipeline.
        question (str): The survey question.
        answer (str): The respondent's answer.
    
    Returns:
        float: The probability score for the label 'relevant'.
    """
    # Return a score of 0 for empty answers
    if pd.isna(answer) or answer.strip() == "":
        return 0
    
    max_length = 512  # Maximum allowed length for the combined input
    combined_input = f"Question: {question} Answer: {answer}"
    
    # Truncate the input if it exceeds the maximum length
    if len(combined_input) > max_length:
        question_part = f"Question: {question} "
        answer_part = f"Answer: {answer}"
        available_space = max_length - len(question_part)
        if available_space > 20:  # Ensure enough space remains for meaningful content
            truncated_input = question_part + answer_part[:available_space]
        else:
            truncated_input = combined_input[:max_length]
    else:
        truncated_input = combined_input
    
    # Run the classifier with candidate labels 'relevant' and 'irrelevant'
    result = classifier(
        truncated_input,
        candidate_labels=["relevant", "irrelevant"],
        multi_label=False
    )
    
    # Extract and return the relevance score for the 'relevant' label
    relevance_score = result["scores"][result["labels"].index("relevant")]
    return relevance_score

def predict_with_distilbert(tokenizer, model, text, max_length=256):
    """
    Uses the DistilBERT model to predict the quality flag for a given text.
    The model returns:
      - 0 for good quality
      - 1 for bad quality
    
    Args:
        tokenizer: The tokenizer associated with the DistilBERT model.
        model: The DistilBERT model for sequence classification.
        text (str): The combined text to evaluate.
        max_length (int): Maximum token length for the model input.
    
    Returns:
        tuple: (predicted_label, confidence) where predicted_label is 0 or 1,
               and confidence is the model's confidence score for that label.
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        # For empty or invalid input, default to bad quality with full confidence
        return 1, 1.0
    
    # Tokenize and prepare input for the model
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: val.cuda() for key, val in inputs.items()}
    
    # Get model prediction without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Determine predicted label (0 = good quality, 1 = bad quality) and its confidence
    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label].item()
    
    return label, confidence

def combine_predictions(distilbert_pred, roberta_relevance, threshold=0.5):
    """
    Combines predictions from the DistilBERT and RoBERTa models to decide the final quality flag.
    The logic is:
      - If DistilBERT is highly confident about bad quality, flag as bad.
      - If RoBERTa finds the response clearly irrelevant (below threshold), flag as bad.
      - Otherwise, if both models indicate a good response, flag as good.
      - Default to flagging for safety in cases of moderate disagreement.
    
    Args:
        distilbert_pred (tuple): (predicted_label, confidence) from DistilBERT.
        roberta_relevance (float): Relevance score from RoBERTa.
        threshold (float): Relevance score threshold for RoBERTa.
    
    Returns:
        int: 1 if flagged as bad quality, 0 if good quality.
    """
    if distilbert_pred[0] == 1 and distilbert_pred[1] > 0.8:
        return 1
    if roberta_relevance < threshold:
        return 1
    if distilbert_pred[0] == 0 and roberta_relevance >= threshold:
        return 0
    return 1

def determine_final_flag(row, weights, relevance_threshold=0.3):
    """
    Determines the final flag for a respondent by computing a weighted sum of predictions
    from each survey question. If the weighted sum falls below a threshold, the entry is flagged.
    
    Args:
        row (pd.Series): A row from the DataFrame containing predictions for each question.
        weights (dict): A dictionary of weights for each question.
        relevance_threshold (float): The threshold below which the entry is flagged.
    
    Returns:
        int: 1 if the entry is flagged, 0 if not flagged.
    """
    weighted_sum = (
        (1 - row.get("Predicted_Q16A", 1)) * weights["Q16A"] +
        (1 - row.get("Predicted_Q16B", 1)) * weights["Q16B"] +
        (1 - row.get("Predicted_Q18_1", 1)) * weights["Q18_1"] +
        (1 - row.get("Predicted_Q18_2", 1)) * weights["Q18_2"] +
        (1 - row.get("Predicted_Q18_3", 1)) * weights["Q18_3"]
    )
    return 1 if weighted_sum < relevance_threshold else 0

# -------------------------------------------------------------------------
# Enhanced Flag Reporting Functions
# -------------------------------------------------------------------------

def generate_rule_report(row):
    """
    Generates a detailed report of rule-based flags (gibberish and profanity)
    for each survey question. Reports include the question identifier and the issue(s).
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        str: A string report summarizing rule-based flag reasons or 'Clean' if none.
    """
    reasons = []
    for col in columns_to_check:
        issues = []
        # Check for gibberish and profanity flags on the answer for each question
        if row.get(f'{col}_gibberish', False):
            issues.append('gibberish')
        if row.get(f'{col}_profanity', False):
            issues.append('profanity')
        if issues:
            # Extract question identifier (e.g., Q16A or Q18_1) for reporting
            q_number = col.split()[0].split('.')[0]
            reasons.append(f"{q_number}({', '.join(issues)})")
    return " | ".join(reasons) if reasons else "Clean"

def generate_ml_report(row):
    """
    Generates a detailed report of ML-based flags for each survey question.
    For each question flagged by ML, the report includes the question identifier
    and the relevance score.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        str: A string summarizing ML-based flag details or an empty string if none.
    """
    ml_flags = []
    for q in columns_to_check:
        # Determine the column prefix based on the question format
        col_prefix = q[:4] if '.' in q else q[:5]
        pred_col = f"Predicted_{col_prefix}"
        score_col = f"Score_{col_prefix}"
        if row.get(pred_col, 0) == 1:
            q_number = q.split()[0].split('.')[0]
            score = row.get(score_col, 0)
            score_text = f"{score:.2f}" if not pd.isna(score) else "N/A"
            ml_flags.append(f"{q_number}(relevance:{score_text})")
    return " | ".join(ml_flags) if ml_flags else ""

def create_combined_flag_source(row):
    """
    Combines rule-based and ML-based flag reports into a single detailed flag source report.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        str: A combined flag source string, or 'Clean' if no issues are flagged.
    """
    rule_report = generate_rule_report(row)
    ml_report = row.get('ml_flag_details', '')
    distilbert_report = row.get('distilbert_flag_details', '')
    
    reports = []
    if rule_report != "Clean":
        reports.append(f"Rule: {rule_report}")
    if ml_report:
        reports.append(f"ML: {ml_report}")
    if distilbert_report:
        reports.append(f"DistilBERT: {distilbert_report}")
        
    if reports:
        return " | ".join(reports)
    else:
        return "Clean"

# -------------------------------------------------------------------------
# Sidebar Settings: UI Controls and Model Loading
# -------------------------------------------------------------------------

# Display header for analysis settings in the sidebar
st.sidebar.header("Analysis Settings")

# Checkboxes to allow user to choose which ML models to use
use_roberta = st.sidebar.checkbox("Use RoBERTa for Relevance Analysis", value=True)
use_distilbert = st.sidebar.checkbox("Use DistilBERT Quality Prediction", value=True)

# Load models only if at least one ML option is enabled
if use_roberta or use_distilbert:
    roberta_classifier, distilbert_tokenizer, distilbert_model = load_ml_models()
else:
    roberta_classifier, distilbert_tokenizer, distilbert_model = None, None, None

# If RoBERTa is enabled, allow user to adjust weights and threshold for each question
if use_roberta:
    st.sidebar.subheader("Question Weights")
    q16a_weight = st.sidebar.slider("Q16A Weight", 0.0, 1.0, 0.35, 0.05)
    q16b_weight = st.sidebar.slider("Q16B Weight", 0.0, 1.0, 0.35, 0.05)
    q18_1_weight = st.sidebar.slider("Q18_1 Weight", 0.0, 1.0, 0.1, 0.05)
    q18_2_weight = st.sidebar.slider("Q18_2 Weight", 0.0, 1.0, 0.1, 0.05)
    q18_3_weight = st.sidebar.slider("Q18_3 Weight", 0.0, 1.0, 0.1, 0.05)
    
    # Normalize weights so that their sum equals 1
    total = q16a_weight + q16b_weight + q18_1_weight + q18_2_weight + q18_3_weight
    weights = {
        "Q16A": q16a_weight / total,
        "Q16B": q16b_weight / total,
        "Q18_1": q18_1_weight / total,
        "Q18_2": q18_2_weight / total,
        "Q18_3": q18_3_weight / total
    }
    
    # Slider to set the relevance threshold for RoBERTa predictions
    relevance_threshold = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, 0.3, 0.05)
else:
    # Default weights if RoBERTa is not used
    weights = {
        "Q16A": 0.35,
        "Q16B": 0.35,
        "Q18_1": 0.1,
        "Q18_2": 0.1,
        "Q18_3": 0.1
    }
    relevance_threshold = 0.3

# If DistilBERT is enabled, allow user to adjust its confidence threshold
if use_distilbert:
    st.sidebar.subheader("DistilBERT Settings")
    distilbert_threshold = st.sidebar.slider("DistilBERT Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# -------------------------------------------------------------------------
# Main App: File Upload and Data Processing
# -------------------------------------------------------------------------

# File uploader widget for Excel files
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    with st.spinner("Loading and processing data..."):
        try:
            # Read the Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)
            
            # Check if the uploaded file is empty
            if df.empty:
                st.error("The uploaded file is empty.")
                st.stop()
                
            # Clean and map column names using the provided column mapping
            new_columns = []
            for col in df.columns:
                # Remove unwanted characters and split extra information if needed
                clean_col = col.replace('}', '').split('...')[0].strip()
                # Check if the cleaned column matches any expected pattern
                for pattern, target in column_mapping.items():
                    if clean_col.startswith(pattern):
                        new_columns.append(target)
                        break
                else:
                    new_columns.append(col)  # Keep original if no match found

            df.columns = new_columns

            # Verify that all required columns exist; if not, attempt to find a close match
            for col in columns_to_check:
                if col not in df.columns:
                    possible_matches = [c for c in df.columns if col.split()[0] in c]
                    if possible_matches:
                        st.warning(f"Column '{col}' not found. Using '{possible_matches[0]}' instead.")
                        df[col] = df[possible_matches[0]]
                    else:
                        st.error(f"Required column '{col}' not found and no suitable replacement found.")
                        st.stop()

            # Remove duplicate rows from the DataFrame
            df = df.drop_duplicates()
            
            # Add a 'Unique ID' column if it doesn't already exist
            if 'Unique ID' not in df.columns:
                df.insert(0, 'Unique ID', df.index.astype(str))
                st.warning("Added Unique ID column based on index")

            # For each question column, create rule-based flag columns for gibberish and profanity detection
            for col in columns_to_check:
                df[f'{col}_gibberish'] = df[col].apply(is_gibberish)
                df[f'{col}_profanity'] = df[col].apply(contains_profanity)
                # Combine the two rule-based flags into a single flag column for each question
                df[f'{col}_flagged'] = df[[f'{col}_gibberish', f'{col}_profanity']].any(axis=1)
            
            # Create an overall rule-based flag if any question was flagged
            flag_columns = [f'{col}_flagged' for col in columns_to_check]
            df['any_rule_flagged'] = df[flag_columns].any(axis=1)
            
            # -----------------------------------------------------------------
            # ML-Based Analysis: RoBERTa Relevance and DistilBERT Quality Prediction
            # -----------------------------------------------------------------
            
            ml_used = False
            distilbert_used = False
            
            # RoBERTa Relevance Analysis
            if use_roberta and roberta_classifier is not None:
                with st.spinner("Running RoBERTa relevance analysis..."):
                    # Prepare a list of question-answer pairs from the DataFrame
                    qa_pairs = []
                    for _, row in df.iterrows():
                        for q in columns_to_check:
                            # Skip if the answer is empty
                            if pd.isna(row[q]) or str(row[q]).strip() == "":
                                continue
                            qa_pairs.append({
                                "question": extract_question_text(q),  # Extract clean question text
                                "answer": row[q],                        # Respondent's answer
                                "original_q": q,                         # Original question column
                                "unique_id": row["Unique ID"]            # Unique identifier for the respondent
                            })

                    qa_df = pd.DataFrame(qa_pairs)
                    
                    if qa_df.empty:
                        st.warning("No valid question-answer pairs found for RoBERTa analysis.")
                    else:
                        # Initialize progress bar for RoBERTa analysis
                        progress_bar = st.progress(0)
                        relevance_scores = []
                        total_rows = len(qa_df)
                        
                        # Compute relevance scores for each question-answer pair
                        for i, (_, row) in enumerate(qa_df.iterrows()):
                            score = get_relevance_score(roberta_classifier, row["question"], row["answer"])
                            relevance_scores.append(score)
                            progress_bar.progress((i + 1) / total_rows)
                        
                        qa_df["relevance_score"] = relevance_scores
                        
                        # Label responses: 0 for relevant (score >= 0.5), 1 for irrelevant (score < 0.5)
                        qa_df["predicted_label"] = qa_df["relevance_score"].apply(lambda x: 0 if x >= 0.5 else 1)
                        
                        # Create mapping dictionaries to map predictions and scores back to the original DataFrame
                        prediction_map = {}
                        score_map = {}
                        for _, row in qa_df.iterrows():
                            key = (row["unique_id"], row["original_q"])
                            prediction_map[key] = row["predicted_label"]
                            score_map[key] = row["relevance_score"]
                        
                        # Map the predictions and scores for each question back to the main DataFrame
                        for q in columns_to_check:
                            # Determine the column prefix based on question formatting
                            if '_' in q:
                                col_prefix = q[:5]
                            else:
                                col_prefix = q[:4]
                            
                            # Function to get prediction for each row based on unique_id and question
                            def get_prediction(row):
                                key = (row["Unique ID"], q)
                                return prediction_map.get(key, 1)  # Default to 1 (irrelevant) if not found
                            
                            # Function to get score for each row
                            def get_score(row):
                                key = (row["Unique ID"], q)
                                return score_map.get(key, np.nan)  # Default to NaN if not found
                            
                            df[f"Predicted_{col_prefix}"] = df.apply(get_prediction, axis=1)
                            df[f"Score_{col_prefix}"] = df.apply(get_score, axis=1)
                        
                        # Calculate the final flag using the weighted predictions and relevance threshold
                        df["Final_RoBERTa_Flag"] = df.apply(
                            lambda row: determine_final_flag(row, weights, relevance_threshold), 
                            axis=1
                        )
                        
                        # Generate a detailed ML flag report for each row
                        df['ml_flag_details'] = df.apply(generate_ml_report, axis=1)
                        ml_used = True
                        
                        # Display RoBERTa analysis statistics in the app
                        st.subheader("RoBERTa Relevance Analysis Results")
                        ml_stats = pd.DataFrame({
                            'Question': columns_to_check,
                            'Good Responses': [df[f"Predicted_{q[:4] if '.' in q else q[:5]}"].eq(0).sum() for q in columns_to_check],
                            'Bad Responses': [df[f"Predicted_{q[:4] if '.' in q else q[:5]}"].eq(1).sum() for q in columns_to_check],
                        })
                        st.dataframe(ml_stats)
            
            # DistilBERT Quality Prediction
            if use_distilbert and distilbert_tokenizer is not None and distilbert_model is not None:
                with st.spinner("Running DistilBERT quality prediction..."):
                    # Combine all answers for a respondent into one text string
                    df['combined_text'] = df[columns_to_check].fillna('').astype(str).agg(' '.join, axis=1)
                    
                    progress_bar = st.progress(0)
                    distilbert_predictions = []
                    distilbert_confidences = []
                    total_rows = len(df)
                    
                    # Predict quality for each respondent's combined text
                    for i, text in enumerate(df['combined_text']):
                        pred, conf = predict_with_distilbert(distilbert_tokenizer, distilbert_model, text)
                        distilbert_predictions.append(pred)
                        distilbert_confidences.append(conf)
                        progress_bar.progress((i + 1) / total_rows)
                    
                    # Store predictions and confidence scores in new DataFrame columns
                    df['DistilBERT_Prediction'] = distilbert_predictions
                    df['DistilBERT_Confidence'] = distilbert_confidences
                    
                    # Generate a detailed flag string for DistilBERT predictions when flagged as bad quality
                    df['distilbert_flag_details'] = df.apply(
                        lambda row: f"Qual({row['DistilBERT_Prediction']}@{row['DistilBERT_Confidence']:.2f})" 
                        if row['DistilBERT_Prediction'] == 1 else "", 
                        axis=1
                    )
                    
                    # Store the final DistilBERT flag directly from the prediction
                    df['Final_DistilBERT_Flag'] = df['DistilBERT_Prediction']
                    distilbert_used = True
                    
                    # Display DistilBERT analysis statistics in the app
                    st.subheader("DistilBERT Quality Prediction Results")
                    distilbert_stats = pd.DataFrame({
                        'Prediction': ['Good Quality', 'Bad Quality'],
                        'Count': [
                            (df['DistilBERT_Prediction'] == 0).sum(),
                            (df['DistilBERT_Prediction'] == 1).sum()
                        ]
                    })
                    st.dataframe(distilbert_stats)
            
            # -----------------------------------------------------------------
            # Combine Flags from Rule-Based and ML-Based Analysis
            # -----------------------------------------------------------------
            
            if ml_used and distilbert_used:
                # Combine using a logical OR between rule-based, RoBERTa, and DistilBERT flags
                df['combined_flag'] = (
                    df['any_rule_flagged'] | 
                    df['Final_RoBERTa_Flag'] | 
                    df['Final_DistilBERT_Flag']
                ).astype(int)
            elif ml_used:
                df['combined_flag'] = (df['any_rule_flagged'] | df['Final_RoBERTa_Flag']).astype(int)
            elif distilbert_used:
                df['combined_flag'] = (df['any_rule_flagged'] | df['Final_DistilBERT_Flag']).astype(int)
            else:
                df['combined_flag'] = df['any_rule_flagged'].astype(int)
                
            # Create a combined flag source report for each respondent
            df['flag_source'] = df.apply(create_combined_flag_source, axis=1)
                
            # -----------------------------------------------------------------
            # Prepare and Display Final Results
            # -----------------------------------------------------------------
            
            # Define the columns to display in the final report
            display_columns = ['Unique ID'] + columns_to_check + ['flag_source', 'combined_flag']
            result_columns = [col for col in display_columns if col in df.columns]
            
            if len(result_columns) < len(display_columns):
                missing_cols = [col for col in display_columns if col not in result_columns]
                st.warning(f"Some display columns are missing: {', '.join(missing_cols)}")
            
            # Create a result DataFrame and add a human-readable status column
            result_df = df[result_columns].copy()
            result_df['Status'] = df['combined_flag'].apply(lambda x: 'Flagged' if x else 'Clean')
            
            # Display overall analysis results and summary statistics
            st.subheader("Analysis Results")
            st.write("Summary:")
            col1, col2, col3 = st.columns(3)
            total_entries = len(result_df)
            flagged_entries = (result_df['Status'] == 'Flagged').sum()
            col1.metric("Total Entries", total_entries)
            col2.metric("Flagged Entries", flagged_entries)
            col3.metric("Clean Entries", total_entries - flagged_entries)
            
            # Display a detailed flag distribution table
            st.write("Flag Distribution:")
            flag_dist = df['flag_source'].value_counts().reset_index()
            flag_dist.columns = ['Source', 'Count']
            st.dataframe(flag_dist)
            
            # Display only the flagged entries for review
            st.subheader("Flagged Entries")
            flagged_df = result_df[result_df['Status'] == 'Flagged'].copy()
            if flagged_df.empty:
                st.info("No entries were flagged.")
            else:
                st.dataframe(
                    flagged_df.style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == 'Flagged' else '',
                        subset=['Status']
                    )
                )
            
            # Display a preview of all entries with flag highlights
            st.subheader("All Entries Preview")
            st.dataframe(
                result_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'Flagged' else '',
                    subset=['Status']
                )
            )
            
            # -----------------------------------------------------------------
            # Export the Analysis Results to Excel and Provide Download Options
            # -----------------------------------------------------------------
            
            full_output_path = "all_entries_with_flags.xlsx"
            result_df.to_excel(full_output_path, index=False)
            
            flagged_output_path = "flagged_entries.xlsx"
            flagged_df.to_excel(flagged_output_path, index=False)
            
            col1, col2 = st.columns(2)
            with open(full_output_path, "rb") as file:
                col1.download_button(
                    label="Download Full Report",
                    data=file,
                    file_name="market_survey_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with open(flagged_output_path, "rb") as file:
                col2.download_button(
                    label="Download Flagged Entries Only",
                    data=file,
                    file_name="flagged_entries.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        except Exception as e:
            # Display error message and exception details if any error occurs during processing
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)