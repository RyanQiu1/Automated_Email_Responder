from datetime import datetime
import win32com.client
import pandas as pd
import csv
import re
import json
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from elasticsearch import Elasticsearch
from flask import Flask, request
from flask_cors import CORS
import pythoncom
from collections import Counter
import pickle
from nltk.stem import PorterStemmer
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
import os
from joblib import dump, load


es = Elasticsearch(['https://localhost:9200'], basic_auth=('elastic',
                   'SSv25TppXxgGOHZq9p5k'),  verify_certs=False)
# Load the tokenizer from the directory containing tokenizer.json
tokenizer = AutoTokenizer.from_pretrained("Final_NER")

# Load the model from the safetensors file
model = AutoModelForTokenClassification.from_pretrained("Final_NER")

# Create the NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer,
               aggregation_strategy="simple")


def extract_incident_category(text):
    # Regex to find "Timestamp {some text}"
    match = re.search(r"Incident category\s+(.*)", text)
    if match:
        return match.group(1)  # Returns the content within the braces
    return None  # Returns None if no match is found


def extract_vulnerability(text):
    # Regex to find "Timestamp {some text}"
    match = re.search(r"Vulnerability information\s+(.*)", text)
    if match:
        return match.group(1)  # Returns the content within the braces
    return None  # Returns None if no match is found


def extract_incident_type(text):
    # Regex to find "Incident type" followed by its description
    match = re.search(r"Incident type\s+(.*)", text)
    if match:
        # Extract the description content
        description = match.group(1).strip()
        # Remove unwanted patterns
        cleaned_description = re.sub(
            r"\[Website Related] If others,", "", description).strip()
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()

        # If the cleaned description is empty, handle alternative pattern
        if not cleaned_description or cleaned_description == "[Website Related] If others,":
            alternative_match = re.search(
                r"\[Malware / Device Related\] Incident type\s+(.*)", text)
            if alternative_match:
                alternative_description = alternative_match.group(1).strip()
                cleaned_description = re.sub(
                    r"[\t\n\r\f\v]+", " ", alternative_description).strip()
        cleaned_description = re.sub(
            r"\[Malware / Device Related\] Infected device", "", cleaned_description).strip()
        return cleaned_description if cleaned_description else None

    return None


def extract_phishing(text):
    # Using re.findall to handle potential multiple matches and making the search case-insensitive
    matches = re.findall(
        r"(Reported website URL/IP|IP/URL)\s+(.*)", text, re.IGNORECASE)
    if matches:
        # Assuming you only want the first match's second group
        first_match_content = matches[0][1]
        # Remove unwanted patterns and clean up whitespace
        cleaned_content = re.sub(
            r"\[Website Related\] Configuration and version of the software/plugin", "", first_match_content)
        cleaned_content = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_content).strip()
        return cleaned_content
    return None


def extract_others(text):
    # Regex to find "Incident Type" followed by its description
    match = re.search(r"\[Website Related\] If others\,\s+(.*)", text)
    if match:
        # Extract the description content
        description = match.group(1).strip()
        # Remove unwanted patterns
        cleaned_description = re.sub(
            r"Vulnerability Information", "", description)
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def extract_description_of_vul(text):
    # Regex to find "Incident Type" followed by its description
    match = re.search(r"Description of vulnerability\s+(.*)", text)
    if match:
        # Extract the description content
        description = match.group(1).strip()
        # Remove unwanted patterns
        cleaned_description = re.sub(r"\[Website Related\]", "", description)
        cleaned_description = re.sub(
            r"Reported website URL\/IP", "", cleaned_description)
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def extract_account_related_additional_information(text):
    # Regex to find "[Account Related] Additional information" followed by its description
    # Using re.DOTALL to match across multiple lines if needed
    match = re.search(
        r"\[Account Related\] Additional information\s+(.*?)(?=\[Case Enquiry\] Case reference number)", text, re.DOTALL)
    if match:
        # Extract the description content and strip to remove leading/trailing whitespace
        description = match.group(1).strip()
        return description
    return None


def is_ransomware(text):
    # Regex to find "[Account Related] Additional information" followed by its description
    # Using re.DOTALL to match across multiple lines if needed
    match = re.search(
        r"What is the ransomware variant\?\s+(.*?)", text, re.DOTALL)
    if match:
        # Extract the description content and strip to remove leading/trailing whitespace
        description = match.group(1).strip()
        return description
    return None


def ransomware_amount(text):
    # Updated Regex to include a possible ending delimiter, making the pattern case-insensitive
    match = re.search(
        r"What is the ransom amount demanded\?\s+(.*)", text, re.IGNORECASE)
    if match:
        # Extract the description content
        description = match.group(1).strip()
        # Remove unwanted patterns
        cleaned_description = re.sub(
            r"\[Ransomware\] Have you made a police report\?", "", description)
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def business_email_compromise(text):
    match = re.search(r"Additional Information[:\s]+(.*)", text, re.IGNORECASE)
    if match:
        # Extract the description content
        description = match.group(1).strip()
        # Remove the specific unwanted pattern, adjust regex for potential variations in spacing/punctuation
        cleaned_description = re.sub(
            r"\[Malware\s*/\s*Device\s*Related\]\s*Incident\s*type", "", description, flags=re.IGNORECASE)
        # Normalize whitespace
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def extract_device_related_additional_information(text):
    # Regex to capture content after "[Malware / Device Related] Additional information"
    match = re.search(
        r"\[Malware \/ Device Related\] Additional information\s+(.*)", text, re.DOTALL)
    if match and match.group(1).strip():
        # Extract the description content
        description = match.group(1).strip()

        # Combine multiple removals into fewer regex operations where possible
        patterns_to_remove = [
            r"\[Account Related - Organisation\] Incident type.*",
            r"\[Account Related - Individual\] Incident type.*",
            r"\[Account Related\].*",
            r"Others.*",
            r"Method of reporting credential leak.*",
            r"Source of credential leak.*",
            r"Are you still able to access your account\?.*",
            r"Additional information.*",
            r"\[Case Enquiry\] Case reference number.*",
            r"\[Case Enquiry\] Your enquiry.*"
        ]
        cleaned_description = description
        for pattern in patterns_to_remove:
            cleaned_description = re.sub(
                pattern, "", cleaned_description, flags=re.IGNORECASE | re.DOTALL)

        # Normalize whitespace to clean up the result after removing the pattern
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def conversation_details(text):
    # Using re.DOTALL to match across multiple lines if needed and re.IGNORECASE for case insensitivity
    # Adjust the regex to capture text immediately following a space after "Conversation details"
    match = re.search(r"Conversation details\s+(.+)",
                      text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        # Extract the description content
        description = match.group(1).strip()

        # Combine multiple removals into fewer regex operations where possible
        patterns_to_remove = [
            r"\[Account Related - Organisation\] Incident type.*",
            r"\[Account Related - Individual\] Incident type.*",
            r"\[Account Related\].*",
            r"Others.*",
            r"Method of reporting credential leak.*",
            r"Source of credential leak.*",
            r"Are you still able to access your account\?.*",
            r"Additional information.*",
            r"\[Case Enquiry\] Case reference number.*",
            r"\[Case Enquiry\] Your enquiry.*",
            r"\[Malware \/ Device Related\].*"
        ]
        cleaned_description = description
        for pattern in patterns_to_remove:
            cleaned_description = re.sub(
                pattern, "", cleaned_description, flags=re.IGNORECASE | re.DOTALL)

        # Normalize whitespace to clean up the result after removing the pattern
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def website_related_additional_info(text):
    # Using re.DOTALL to match across multiple lines if needed and re.IGNORECASE for case insensitivity
    # Adjust the regex to capture text immediately following a space after "Conversation details"
    match = re.search(
        r"\[Website Related\] Additional information\s+(.+)", text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        # Extract the description content
        description = match.group(1).strip()

        # Combine multiple removals into fewer regex operations where possible
        patterns_to_remove = [
            r"\[Account Related - Organisation\] Incident type.*",
            r"\[Account Related - Individual\] Incident type.*",
            r"\[Account Related\].*",
            r"Others.*",
            r"Method of reporting credential leak.*",
            r"Source of credential leak.*",
            r"Are you still able to access your account\?.*",
            r"Additional information.*",
            r"\[Case Enquiry\] Case reference number.*",
            r"\[Case Enquiry\] Your enquiry.*",
            r"\[Malware \/ Device Related\].*"
        ]
        cleaned_description = description
        for pattern in patterns_to_remove:
            cleaned_description = re.sub(
                pattern, "", cleaned_description, flags=re.IGNORECASE | re.DOTALL)

        # Normalize whitespace to clean up the result after removing the pattern
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def case_enquiry(text):
    # Using re.DOTALL to match across multiple lines if needed and re.IGNORECASE for case insensitivity
    # Adjust the regex to capture text immediately following a space after "Conversation details"
    match = re.search(r"\[Case Enquiry\] Your enquiry\s+(.+)",
                      text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        # Extract the description content
        description = match.group(1).strip()

        # Combine multiple removals into fewer regex operations where possible
        patterns_to_remove = [
            r"\[Account Related - Organisation\] Incident type.*",
            r"\[Account Related - Individual\] Incident type.*",
            r"\[Account Related\].*",
            r"Others.*",
            r"Method of reporting credential leak.*",
            r"Source of credential leak.*",
            r"Are you still able to access your account\?.*",
            r"Additional information.*",
            r"\[Case Enquiry\] Case reference number.*",
            r"\[Malware \/ Device Related\].*"
        ]
        cleaned_description = description
        for pattern in patterns_to_remove:
            cleaned_description = re.sub(
                pattern, "", cleaned_description, flags=re.IGNORECASE | re.DOTALL)

        # Normalize whitespace to clean up the result after removing the pattern
        cleaned_description = re.sub(
            r"[\t\n\r\f\v]+", " ", cleaned_description).strip()
        return cleaned_description
    return None


def extract_keywords(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return [""] * 5  # Return empty list of 5 elements if text is empty
    prompt = f"Can you give 5 words related to cyber security for this text, give it in an array: {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )
    if response.status_code == 200:
        # Assuming the response structure you provided
        response_data = response.json()
        # Directly extracting keywords as the response is expected to be a JSON list within 'content'
        try:
            keywords_text = response_data['choices'][0]['message']['content']
            # Extracting the array part from the given string, assuming format is fixed
            start = keywords_text.find('[')
            end = keywords_text.find(']') + 1
            keywords = json.loads(keywords_text[start:end])
            return keywords + [""] * (5 - len(keywords))  # Ensuring 5 elements
        except json.JSONDecodeError:
            return [""] * 5
    else:
        return [""] * 5
    

def extract_account_phrase(text):
    """
    Make an API call to the local language model to extract up to 5 keywords summarizing the incident described in the text.
    """
    if not text:
        return ""

    # Define a more streamlined and direct prompt
    prompt = f"Summarize the incident in 5 words or fewer: {text}"

    # Set up the request to the language model
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )

    # Process the response
    if response.status_code == 200:
        # Extracting the summarized content directly from the response
        try:
            return response.json()['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            print("Error parsing the response.")
            return ""
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""

# Function to extract keywords from the local LLM


def extract_does_involve_money(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Does this incident involve the user losing money (just output yes or no): {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""

# Function to extract keywords from the local LLM


def extract_type_of_device(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"can give me the type of device involve in this incident (like, phone, router, email, sms, laptop, social media, messanger app, call, etc), (output max 5 words): {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""


def get_template_response(category):

    # Define the index and queries
    index = 'email_templates'
    queries = [
        {"match": {"subject": "Starting Header"}},
        {"match": {"subject": f"{category}"}},
        {"match": {"subject": "Ending Header"}},
        # Add more queries as needed
    ]

    # Initialize an empty list to store bodies
    bodies = []

    # Retrieve and store bodies of each query result
    for query in queries:
        # Perform the search query
        response = es.search(index=index, body={"query": query, "size": 1})

        # Append the body of the first result to the list
        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            bodies.append(hit['_source'].get('body', 'N/A'))

    # Combine bodies into a single string
    combined_body = '\n'.join(bodies)

    # Print the combined body
    return combined_body


def load_ner_model(text):
    # Make dictionary to store the entities
    # Get the NER results
    ner_results = nlp(text)

    # Initialize a dictionary to store words by entity groups of interest
    entity_words = {}

    # Define the entity groups you are interested in
    interested_entities = ['MEDIA', 'ARTIFACT']

    # Loop through the NER results
    for result in ner_results:
        entity_group = result['entity_group']
        word = result['word']

        # Check if the entity group is one of the interested ones
        if entity_group in interested_entities:
            # Initialize the list for the entity group if not already initialized
            if entity_group not in entity_words:
                entity_words[entity_group] = []
            # Append the word to the appropriate list
            entity_words[entity_group].append(word)
    return entity_words


def categorize_sub_cat(sub_cat_dict):
    # Define keywords for each category
    category_keywords = {
        'Social Messaging App': ['telegram', 'whatsapp', 'signal', 'messenger', 'wechat', 'viber', 'line', 'kakao', 'discord', 'slack', 'zoom', 'skype', 'teams', 'hangouts', 'chat', 'messaging', 'video call'],
        'Phone': ['phone', 'mobile', 'cell', 'smartphone', 'iphone', 'android', 'samsung', 'nokia', 'huawei', 'xiaomi', 'oneplus'],
        'Router': ['router', 'modem', 'wifi', 'internet', 'network', 'ethernet', 'lan', 'wan'],
        'Email': ['email', 'gmail', 'yahoo', 'outlook', 'hotmail', 'protonmail', 'aol', 'icloud'],
        'Computer': ['computer', 'laptop', 'desktop', 'pc', 'mac', 'windows', 'linux', 'chromebook', 'surface', 'thinkpad', 'macbook'],
        'Account': ['account', 'login', 'credential', 'password', 'username', 'access', 'authentication', 'authorization', 'verification'],
        'SMS': ['sms', 'message'],
        'Messenger Application/Social Media Account': ['instagram', 'facebook', 'twitter', 'linkedin', 'snapchat', 'tiktok', 'pinterest', 'reddit', 'youtube', 'social', 'media'],
        'Website': ['website', 'web', 'site', 'url', 'link', 'webpage', 'domain', 'internet']
    }

    # Initialize sub_cat as None to indicate no category found initially
    sub_cat = ""

    # Ensure sub_cat_dict is provided and not empty
    if sub_cat_dict:
        # Iterate over each category and its keywords
        for category, keywords in category_keywords.items():
            # Check both 'MEDIA' and 'ARTIFACT' fields if available
            for words in keywords:
                if 'MEDIA' in sub_cat_dict and words in sub_cat_dict['MEDIA'][0]:
                    sub_cat = category
                    return sub_cat

                if 'ARTIFACT' in sub_cat_dict and words in sub_cat_dict['ARTIFACT'][0]:
                    sub_cat = category
                    return sub_cat

    return sub_cat


def call_to_llm(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Does this involve the company or person (output one word): {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""


def call_to_llm_is_cyber_related(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Cybersecurity means Phishing, extortion, scam emails (send us your email header at our Report a Phishing Email initiative), Phishing websites, Ransomware attacks, Website defacements, Malware hosting/Command and Control Servers, attempts (either failed or successful) to disrupt or gain access to a network, system or its data. Thus is this cyber security related? (return yes or no only): {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            # "tokens_to_generate": 1
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""


def call_to_llm_is_ald_related(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Is this text related to harassment/extortion, social media impersonation, phishing, scam, cyber crime, or money lost? (return yes or no only): {text}?"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            # "tokens_to_generate": 1
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""
    
def call_to_llm_is_ald_related(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Given the text: '{text}', determine which category it belongs to. Choose from the following categories: Blackmail, Brute Force Attack, Business Email Compromised, Case Enquiry, Compromised, Cyber Crime, Cyber Crime - Reported to police, Defacement, Email Notification of Data Breaches, Extortion Email, Fake, Possible Scam or Impersonation Website, Harassment, Insufficient Information, Malware Hosting App, Non-Cybersecurity Related Reports, Phishing, Possible Fake/Impersonation Sites, Ransomware, Scam/Gambling/Investment/Unlicensed Money Lending/Pornography Sites, Social Media Impersonation, Spam Email, Spoofed Email, Spyware, Tech Support Scam, Vishing."
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            # "tokens_to_generate": 1
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""
    
def llm_summarise(text):
    """Make an API call to the local language model to extract keywords."""
    if text == "":
        return ""
    prompt = f"Please summarize the cybersecurity incident described in the text below in one paragraph. Include the names of any individuals involved, types of devices affected, and a concise summary of the incident: {text}"
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Authorization": "Bearer lm-studio"},
        json={
            "model": "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            # "tokens_to_generate": 1
        }
    )
    if response.status_code == 200:
        # Assuming the response is structured with the keywords directly in 'content'
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""


class EmailAutomationApp:
    def __init__(self):
        self.unread_emails = []
        self.category = []
        self.draft_emails = []
        self.loaded_model = None
        self.loaded_vectorizer = None
        self.loaded_label_encoder = None
        self.using_local_model = False
        self.email = ''

    def pre_process_saved_csv(self):
        # Load the CSV, apply regex extraction, and save to a new file
        df = pd.read_csv('email_export.csv', encoding='latin1')
        df['Incident category'] = df['Email Body'].apply(
            lambda x: extract_incident_category(x))
        df['Vulnerability Information'] = df['Email Body'].apply(
            lambda x: extract_vulnerability(x))
        df['Incident Type'] = df['Email Body'].apply(
            lambda x: extract_incident_type(x))
        df['Reported website URL/IP'] = df['Email Body'].apply(
            extract_phishing)
        df['Others'] = df['Email Body'].apply(extract_others)
        df['Description of vulnerability'] = df['Email Body'].apply(
            extract_description_of_vul)
        df['Account Related Information'] = df['Email Body'].apply(
            extract_account_related_additional_information)
        df['Ransomware Variant'] = df['Email Body'].apply(is_ransomware)
        df['Ransomware Amount'] = df['Email Body'].apply(ransomware_amount)
        df['Additional Information'] = df['Email Body'].apply(
            business_email_compromise)
        df['Device Related Information'] = df['Email Body'].apply(
            extract_device_related_additional_information)
        df['Conversation Details'] = df['Email Body'].apply(
            conversation_details)
        df['Website Related Additional Info'] = df['Email Body'].apply(
            website_related_additional_info)
        df['Case Enquiry'] = df['Email Body'].apply(case_enquiry)

        df['Keywords'] = df['Additional Information'].apply(
           lambda text: extract_keywords(
               text.strip()) if text.strip() else [""] * 5
        )

        df[['1st Column', '2nd Column', '3rd Column', '4th Column', '5th Column']
          ] = pd.DataFrame(df['Keywords'].tolist(), index=df.index)

        # Optionally, drop the temporary 'Keywords' column if no longer needed
        df.drop('Keywords', axis=1, inplace=True)
        df['Account Related Information'] = df['Account Related Information'].fillna(
            "")
        df['Website Relation Information'] = df['Website Related Additional Info'].fillna(
            "")
        df['Device Related Information'] = df['Device Related Information'].fillna(
            "")
        df['Phrase from Account Related'] = df['Account Related Information'].apply(
            lambda text: extract_account_phrase(text) if text and text.strip() != "" else "")
        df['Phrase for Website Related information'] = df['Website Related Additional Info'].apply(
            lambda text: extract_account_phrase(text) if text and text.strip() != "" else "")
        df['Phrase for Device Related Information'] = df['Device Related Information'].apply(
            lambda text: extract_account_phrase(text) if text and text.strip() != "" else "")
        df['Does this incident involve money'] = df['Additional Information'].apply(
            lambda text: extract_does_involve_money(text) if text and text.strip() != "" else "")
        df['Does this incident involve money'] = df['Account Related Information'].apply(
            lambda text: extract_does_involve_money(text) if text and text.strip() != "" else "")
        df['Does this incident involve money'] = df['Device Related Information'].apply(
            lambda text: extract_does_involve_money(text) if text and text.strip() != "" else "")
        df['Does this incident involve money'] = df['Website Related Additional Info'].apply(
            lambda text: extract_does_involve_money(text) if text and text.strip() != "" else "")
        df.to_csv('extracted_test_data_1.csv', index=False)
        return True

    def export_email_as_csv(self, email_index):
        """Exports the selected email as a CSV file with headers."""
        if 0 <= email_index < len(self.unread_emails):
            email = self.unread_emails[email_index]
            with open('email_export.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the headers
                writer.writerow(['Subject', 'Email Body'])
                # Write the email subject and body
                writer.writerow([email['subject'], email['body']])

            result = self.pre_process_saved_csv()
            if result:
                return {"message": "Email exported successfully"}
            else:
                return {"error": "Failed to export email"}
        else:
            return {"error": "Invalid email index"}
        
    def get_email_accounts(self):
        """Fetches email accounts from Outlook."""
        outlook = win32com.client.Dispatch(
            "Outlook.Application", pythoncom.CoInitialize()).GetNamespace("MAPI")
        accounts = outlook.Folders
        email_accounts = [account.Name for account in accounts]
        print(email_accounts)
        return json.dumps(email_accounts)

    def get_unread_emails(self, email):
        """Fetches unread emails from Outlook inbox."""
        outlook = win32com.client.Dispatch(
            "Outlook.Application", pythoncom.CoInitialize()).GetNamespace("MAPI")
        accounts = outlook.Folders

        self.email = email

        for account in accounts:
            if account.Name == email:
                inbox = account.Folders["Inbox"]
                messages = inbox.Items
                messages.Sort("[ReceivedTime]", True)
                self.unread_emails = [
                    {
                        "id": index + 1,
                        "subject": msg.Subject,
                        "sender": msg.SenderEmailAddress,
                        "receivedAt": msg.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S"),
                        "body": msg.Body
                    } for index, msg in enumerate(messages) if msg.UnRead
                ]
                return json.dumps(self.unread_emails)

    def get_draft_emails(self, email):
        """Fetches draft emails from Outlook inbox."""
        outlook = win32com.client.Dispatch(
            "Outlook.Application", pythoncom.CoInitialize()).GetNamespace("MAPI")
        accounts = outlook.Folders

        self.email = email
        for account in accounts:
            if account.Name == email:
                inbox = account.Folders["Drafts"]
                messages = inbox.Items
                messages.Sort("[ReceivedTime]", True)
                print(messages)
                self.draft_emails = [
                    {
                        "id": index + 1,
                        "subject": msg.Subject,
                        "sender": msg.SenderEmailAddress,
                        "receivedAt": msg.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S"),
                        "body": msg.Body
                    } for index, msg in enumerate(messages) if msg
                ]
                return json.dumps(self.draft_emails)

    def save_draft_response(self, index, reply_body, account_name):
        try:
            # Initialize Outlook application
            outlook = win32com.client.Dispatch(
                "Outlook.Application", pythoncom.CoInitialize())
            namespace = outlook.GetNamespace("MAPI")

            # Find the specific account
            account = None
            for acc in namespace.Folders:
                if acc.Name == account_name:
                    account = acc
                    break

            if not account:
                return {"error": "Account not found"}

            # Get the inbox folder of the specific account
            inbox = account.Folders("Inbox")

            # Get all messages in the inbox
            messages = inbox.Items

            # Ensure index is within valid range
            if 1 <= int(index) <= messages.Count:
                # Get the message at the specified index (Outlook's collection is 1-based)
                message = messages.Item(index)

                # Reply to the message
                reply = message.Reply()
                reply.Subject = f"Re: {message.Subject}"
                reply.Body = reply_body

                # Save the reply as a draft
                reply.Save()

                return {"message": "Email replied and saved as draft successfully"}
            else:
                return {"error": "Invalid index provided"}

        except Exception as e:
            return {"error": str(e)}
        

    def save_existing_draft_response(self, email_index, content, email_address):
        try: 
            # Initialize Outlook application
            outlook = win32com.client.Dispatch("Outlook.Application", pythoncom.CoInitialize())
            namespace = outlook.GetNamespace("MAPI")

            # Find the specific account
            account = None
            for acc in namespace.Folders:
                if acc.Name == email_address:
                    account = acc
                    break

            if not account:
                return {"error": "Account not found"}

            # Get the drafts folder of the specific account
            drafts = account.Folders("Drafts")

            # Get all messages in the drafts folder
            messages = drafts.Items

            # Ensure index is within valid range
            if 1 <= int(email_index) <= messages.Count:
                # Get the message at the specified index (Outlook's collection is 1-based)
                message = messages.Item(email_index)

                # Update the body of the existing draft
                message.Body = content

                # Save the updated draft
                message.Save()

                return {"message": "Email draft updated successfully"}
            else:
                return {"error": "Invalid index provided"}
        
        except Exception as e:
            return {"error": str(e)}
        

    def load_model_and_predict(self):
        if self.using_local_model:
            data = pd.read_csv('extracted_test_data_1.csv', encoding='latin1')
            # extract the text data from the specified columns
            category = call_to_llm_is_ald_related(data['Email Body'][0])
            template = get_template_response(category)
            return json.dumps({'category': category, 'template': template})
            
        else:
            """Loads the trained model and predicts the target variable."""
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

            # Load NLTK stopwords
            stop_words = set(stopwords.words('english'))

            data = pd.read_csv('extracted_test_data_1.csv', encoding='latin1')
            stemmer = PorterStemmer()

            def preprocess_text(text):
                tokens = word_tokenize(text.lower())
                tokens = [stemmer.stem(
                    token) for token in tokens if token.isalpha() and token not in stop_words]
                return ' '.join(tokens)

            # Specify columns to combine into a single text field
            included_columns = [
                'Description of vulnerability',
                'Website Related Additional Info',
                'Others',
                'Account Related Information',
                'Additional Information', 'Case Enquiry',
                'Device Related Information',
                'Phrase for Website Related information',
                'Phrase for Device Related Information', 'Phrase from Account Related', 'Conversation Details', 'Ransomware Variant', 'Ransomware Amount',
                'Incident category', 'Incident Type', 'Reported website URL/IP',
                '1st Column', '2nd Column', '3rd Column', '4th Column', '5th Column', 'Does this incident involve money'
            ]

            self.loaded_model = load('models/final_ensemble_model.joblib')

            # Load the TF-IDF vectorizer
            self.loaded_vectorizer = load('models/final_tfidf_vectorizer.joblib')

            # Load the label encoder
            self.loaded_label_encoder = load('models/final_label_encoder.joblib')

            # Combine the text data from the specified columns
            data['combined_text'] = data[included_columns].fillna(
                '').agg(' '.join, axis=1)
            data['processed_text'] = data['combined_text'].apply(preprocess_text)

            # Vectorize the new processed text using the loaded TF-IDF vectorizer
            new_tfidf_features = self.loaded_vectorizer.transform(
                data['processed_text'])

            # Predict using the loaded model
            new_probabilities = self.loaded_model.predict_proba(new_tfidf_features)

            # Ensure the sorting gives the right shape:
            top3_classes = np.argsort(-new_probabilities, axis=1)[:, :3]
            top3_probs = -np.sort(-new_probabilities, axis=1)[:, :3]

            # Decoding check:
            top3_class_names = self.loaded_label_encoder.inverse_transform(
                top3_classes.flatten()).reshape(top3_classes.shape)
            array_for_prediction = []

            # Displaying results
            for i, (classes, probs) in enumerate(zip(top3_class_names, top3_probs)):
                for class_name, prob in zip(classes, probs):
                    array_for_prediction.append(class_name)
            self.category = array_for_prediction
            category = array_for_prediction[0]

            def sub_category_ner(text):
                entities = None
                if isinstance(text, str) and text is not None:
                    entities = load_ner_model(text.lower().strip())
                return entities

            sub_cat_dict = {}
            sub_cat = None

            category = category.strip()
            if data['Account Related Information'][0] is not None:

                if call_to_llm_is_cyber_related(data['Account Related Information'][0]).lower() == 'no':
                    if call_to_llm_is_ald_related(data['Account Related Information'][0]).lower() == 'no':
                        category = 'Non-Cyber Security Related Incident'
                        template = get_template_response(category)
                        return json.dumps({'category': category, 'template': template})

            if data['Incident Type'][0] in ['Phishing Website']:
                category = 'Phishing'
                template = get_template_response(category + ' Website')
                return json.dumps({'category': category, 'sub_category': 'Website', 'template': template})

            if isinstance(data['Incident category'][0], str):
                if data['Incident category'][0].strip() in ['Case Enquiry']:
                    category = 'Case Enquiry'
                    template = get_template_response(category)
                    return json.dumps({'category': category, 'template': template})

            if data['Incident Type'][0] in ['Data Breach'] or data['Incident category'][0] in ['Data Breach'] or data['1st Column'][0] in ['Breach'] or data['2nd Column'][0] in ['Breach'] or data['3rd Column'][0] in ['Breach']:
                category = 'Data Breach'
                template = get_template_response(category)
                return json.dumps({'category': category, 'template': template})

            if category in ['Phishing', 'Compromised']:
                if sub_cat_dict == {} and data["Account Related Information"][0] is not None:
                    sub_cat_dict = sub_category_ner(
                    data['Account Related Information'][0])
                elif sub_cat_dict == {} and data["Device Related Information"][0] is not None:
                    sub_cat_dict = sub_category_ner(
                        data['Device Related Information'][0])
                elif sub_cat_dict == {} and data["Website Related Additional Info"][0] is not None:
                    sub_cat_dict = sub_category_ner(
                        data['Website Related Additional Info'][0])
                elif sub_cat_dict == {} and data["Additional Information"][0] is not None:
                    sub_cat_dict = sub_category_ner(
                        data['Additional Information'][0])

                if 'Website' in data["Incident category"][0]:
                    sub_cat = 'Website'

                if 'Business Email Compromise' in data["Incident category"][0]:
                    sub_cat = ''
                    category = 'Business Email Compromise'

                if 'Ransomware' in data["Incident category"][0]:
                    sub_cat = ''
                    category = 'Ransomware'

                if sub_cat_dict:
                    sub_cat = categorize_sub_cat(sub_cat_dict)

                # If sub_cat is not one of the predefined categories, check other phrases
                if sub_cat is None or sub_cat not in ['Phone', 'Router', 'Email', 'Computer', 'Account', 'Social Messaging App', 'SMS', 'Website', 'Social Media']:
                    # Initialize list for sub_categories
                    sub_cats = []

                    # Check each key and add to sub_cats if key exists and is not empty
                    keys_to_check = [
                        'Phrase for Website Related information',
                        'Phrase for Device Related Information',
                        'Phrase from Account Related',
                    ]

                    for key in keys_to_check:
                        if sub_category_ner(str(data[key][0])) is None:
                            sub_cats.append('')
                        else:
                            sub_cats.append(categorize_sub_cat(sub_category_ner(
                                str(data[key][0]))))

                    # Count the frequency of each sub_category
                    sub_cat_counter = Counter(sub_cats)

                    # Get the most common sub_category
                    if sub_cat_counter:
                        most_common_sub_cat, _ = sub_cat_counter.most_common(1)[0]
                        sub_cat = most_common_sub_cat
                if sub_cat:
                    template = get_template_response(category + ' ' + sub_cat)
                else:
                    template = get_template_response(category)

                return json.dumps({'category': category, 'sub_category': sub_cat, 'template': template})

            else:
                template = get_template_response(category)
                return json.dumps({'category': category, 'template': template})


app = Flask(__name__)
CORS(app)

email_app = EmailAutomationApp()


@app.route('/')
def default_page():
    return json.dumps({'data': 'Welcome to my API'}), 200


@app.route('/get_unread_emails')
def fetch_unread_emails():
    """Returns unread emails as JSON."""
    email = request.args.get('email')  # Get the email query parameter
    unread_emails = email_app.get_unread_emails(email)
    return json.dumps(unread_emails)


@app.route('/load_model_and_predict', methods=['GET'])
def load_model_and_predict():
    """Loads the trained model and predicts the target variable."""
    # if exported once, no need to export again
    # email_app.export_email_as_csv()
    # email_app.pre_process_saved_csv()
    email_app.get_unread_emails(email_app.email)
    email_index = request.args.get('email_index', default=0, type=int)

    # Export the specified email
    export_result = email_app.export_email_as_csv(email_index-1)
    if "error" in export_result:
        return json.dumps(export_result), 400
    else:
        # Perform preprocessing
        # Load model and predict
        prediction = email_app.load_model_and_predict()

        return json.dumps(prediction)


@app.route('/save_draft_response', methods=['POST'])
def save_draft_response():
    """Saves the edited draft email response to Outlook drafts folder."""
    email_index = request.args.get('email_index', default=1, type=int) + 1

    # Get result from the response
    data = request.get_json()
    content = data.get('content')

    if email_index is not None:
        result = email_app.save_draft_response(
            email_index, content, email_app.email)
        if "error" in result:
            return result, 500
        return result, 200
    else:
        return json.dumps({"error": "Index or reply_body not provided"}), 400


@app.route('/save_existing_draft_email', methods=['POST'])
def save_existing_draft_email():
    """Saves the edited draft email response to Outlook drafts folder."""
    data = request.get_json()
    content = data.get('content')
    email_index = request.args.get('email_index', default=1, type=int) + 1

    if email_index and content:
        result = email_app.save_existing_draft_response(
            email_index, content, email_app.email)
        if "error" in result:
            return result, 500
        return result, 200
    else:
        return json.dumps({"error": "Index or reply_body not provided"}), 400
    


@app.route('/save_designation', methods=['POST'])
def save_designation():
    """Saves the edited draft email response to Outlook drafts folder."""
    data = request.get_json()
    content = data.get('name')

    if content:
        try:
            # Define the index and queries
            index = 'email_templates'
            query = {"match": {"subject": "Ending Header"}}  # Example query

            # Perform the search query
            response = es.search(index=index, body={"query": query, "size": 1})

            # Check if document(s) found
            if response['hits']['hits']:
                hit = response['hits']['hits'][0]
                document_id = hit['_id']
                current_body = hit['_source'].get('body', '')

                # Update the document
                updated_document = {
                    'body': '\nRegards, \n' + content  # Update the 'body' field with new content
                }

                es.update(index=index, id=document_id,
                          body={'doc': updated_document})

                # Return a JSON response
                return json.dumps({"message": f"Document with id '{document_id}' updated successfully.",
                                "current_body": current_body,
                                "updated_body": content}), 200
            else:
                return json.dumps({"error": "Document not found."}), 404

        except Exception as e:
            return json.dumps({"error": f"Error updating document: {str(e)}"}), 500

    else:
        return json.dumps({"error": "No content provided."}), 400
    

@app.route('/get_email_templates', methods=['GET'])
def get_email_templates():
    """Returns email templates as JSON."""
    # Define the index
    index = 'email_templates'
    
    # Define a query to exclude specific headers
    query = {
        "bool": {
            "must_not": [
                {"match": {"subject": "Starting Header"}},
                {"match": {"subject": "Ending Header"}},
                {"match": {"subject": "Alpha"}}
            ]
        }
    }

    # Initialize an empty list to store the results
    templates = []

    # Perform the search query
    response = es.search(index=index, body={"query": query, "size": 1000})

    # Retrieve and store subjects and bodies of each query result
    for hit in response['hits']['hits']:
        source = hit['_source']
        subject = source.get('subject', 'N/A')
        body = source.get('body', 'N/A')
        templates.append({'subject': subject, 'body': body})

    # Return the list of templates as JSON
    return json.dumps({"email_templates": templates}), 200


@app.route('/add_email_template', methods=['POST'])
def add_email_template():
    """Adds a new email template to the database."""
    data = request.json
    # Add field to the json data
    index = 'email_templates'
    response = es.index(index=index, body=data)
    if response.get('result') == 'created':
        return json.dumps({'success': True}), 200
    else:
        return json.dumps({'error': 'Error adding document'}), 500


@app.route('/update_email_template', methods=['POST'])
def update_email_template():
    """Updates an existing email template based on the subject."""
    data = request.get_json()
    subject = data.get('subject')

    if not subject:
        return json.dumps({'error': 'Subject is required'}), 400

    index = 'email_templates'
    search_query = {
        "query": {
            "match": {
                "subject": subject
            }
        }
    }

    try:
        search_response = es.search(index=index, body=search_query)
        if not search_response['hits']['hits']:
            return json.dumps({'error': 'Template not found'}), 404

        doc_id = search_response['hits']['hits'][0]['_id']
        update_response = es.update(index=index, id=doc_id, body={"doc": data})
        
        # No need to convert to dict if it's already a dictionary
        return json.dumps('Done'), 200

    except Exception as e:
        # Direct use of jsonify to handle the exception message
        return json.dumps({'error': f'Error updating document: {str(e)}'}), 500
    

@app.route('/get_another_prediction', methods=['GET'])
def get_another_prediction():
    """Loads the trained model and predicts the target variable."""

    prediction_index = request.args.get(
        'prediction_email_index', default=0, type=int)
    next_cat = email_app.category[prediction_index]

    first_cat_response = get_template_response(next_cat)
    return json.dumps({"second_category": next_cat, "second_template": first_cat_response}), 200


@app.route('/get_draft_emails', methods=['GET'])
def get_draft_emails():
    """Returns draft emails as JSON."""
    email = request.args.get('email') 
    draft_emails = email_app.get_draft_emails(email)
    return draft_emails


@app.route('/get_models', methods=['GET'])
def get_models():
    """Returns all .pkl model files in the models directory."""
    # Path to the models directory
    models_dir = './models'
    # Regex pattern to match files ending with .pkl
    pattern = re.compile(r".*\.joblib$")

    # List files in the directory that match the regex
    matched_files = [f for f in os.listdir(models_dir) if pattern.match(f)]

    if matched_files:
        # Return the names of models that match the pattern
        return json.dumps({"models": matched_files}), 200
    else:
        return json.dumps({"error": "No matching model found."}), 404
    


@app.route('/get_email_summary', methods=['GET'])
def get_email_summary():
    email_id = request.args.get('email_id', default=None, type=int) - 1
    print(email_id)
    if email_id is not None and email_id < len(email_app.unread_emails):
        email = email_app.unread_emails[email_id]
        print(email)
        summary = llm_summarise(email['body'])
        return json.dumps({"summary": summary}), 200
    else:
        return json.dumps({"error": "Email not found or invalid email ID"}), 404

    

    
# handle change model
@app.route('/change_model', methods=['POST'])
def change_model():
    """Changes the model used for prediction."""
    data = request.get_json()
    model_name = data.get('model_name')
    if model_name:
        if model_name == 'Local_LLM':
            email_app.using_local_model = True
            return json.dumps({"message": "Model changed to Local_LLM"}), 200
        else:
            email_app.using_local_model = False
            with open('models/final_14_random_forest_classifier.pkl', 'rb') as model_file:
                email_app.loaded_model = pickle.load(model_file)
            with open('models/final_14_tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
                email_app.loaded_vectorizer = pickle.load(vectorizer_file)
            with open('models/final_14_label_encoder.pkl', 'rb') as f:
                email_app.loaded_label_encoder = pickle.load(f)
            return json.dumps({"message": "Model changed to Random Forest Classifier"}), 200
    
    else:
        return json.dumps({"error": "No model name provided."}), 400
    

@app.route('/get_email_accounts', methods=['GET'])
def get_email_accounts():
    """Returns email accounts from Outlook."""
    email_accounts = email_app.get_email_accounts()
    return email_accounts



@app.route('/save_email_in_db', methods=['POST'])
def save_email():
    """Saves the email in the database."""
    data = request.get_json()
    email = data.get('email')

    if email:
        try:
            # Define the index and queries
            index = 'email_templates'
            query = {"match": {"subject": "Alpha"}}  # Example query

            # Perform the search query
            response = es.search(index=index, body={"query": query, "size": 1})

            # Check if document(s) found
            if response['hits']['hits']:
                hit = response['hits']['hits'][0]
                document_id = hit['_id']
                current_body = hit['_source'].get('body', '')

                # Update the document
                updated_document = {
                    'body': email  # Update the 'body' field with new content
                }

                update_response = es.update(index=index, id=document_id,
                                            body={'doc': updated_document})

                # Return a JSON response
                return json.dumps({
                    "message": f"Document with id '{document_id}' updated successfully.",
                    "current_body": current_body,
                    "updated_body": email,
                    "result": update_response.get('result', 'unknown')
                }), 200

            else:
                # Add the email to the database
                create_response = es.index(index=index, body={
                    "body": email,
                    "subject": "Alpha",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                })

                return json.dumps({
                    "message": "New document created successfully.",
                    "document_id": create_response.get('_id')
                }), 201

        except Exception as e:
            return json.dumps({"error": f"Error updating document: {str(e)}"}), 500

    else:
        return json.dumps({"error": "No content provided."}), 400
    

@app.route('/check_email_address', methods=['GET'])
def check_email_address():
    """Checks if the email address is in the database."""
    try:
        # Define the index and queries
        index = 'email_templates'
        search_query = {
            "query": {
                "match": {
                    "subject": 'Alpha'
                }
            }
        }


        search_response = es.search(index=index, body=search_query)



        # Check if document(s) found
        if search_response['hits']['hits']:
            return json.dumps({"message": "Email address found in the database.", "exists": True, "email": search_response['hits']['hits'][0]['_source'].get('body', '')}), 200
        else:
            return json.dumps({"message": "Email address not found in the database.", "exists": False, "email": ""}), 404

    except Exception as e:
        return json.dumps({"error": f"Error checking email address: {str(e)}"}), 500


    

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    

if __name__ == '__main__':
    app.run(debug=True)
  
