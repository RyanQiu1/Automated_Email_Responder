# Email Automation Program

This program is designed to classify emails from the public and retrieve appropriate responses from a database. It assists cybersecurity responders by quickly drafting emails, allowing them to focus on more critical tasks without the need to manually read and respond to each email.

Click [here](UserGuide.md) for the user guide.


# How to start

1. git clone my repo

2. Ask me for the Models, the ensemble of machine learning models and the NER model, create a folder models and model in flask_backend and llm_local folders respectively.

3. Create a virtual enviroment for flask_backend ```flask-backend ```

4. Create a virtual environment for llm_local folder named ```llm_local```

## Download all the packages in requirements.txt in virtual environment
    
    ### Commands

    1. Do these for both flask_backend and llm_local

    2. pip install venv

    3. Run the folder .\venv\scripts\activate

    4. pip install -r requirements.txt

    5. Check All requirements are installed

## To start up (in one command)

1. Start docker services ```sudo systemctl start docker```

2. Change directory to ```cd front-end```, then you can run ```npm start```

3. Wait for loading/or refresh if anything does not show

## Folders

1. flask-backend

    Contains the flask backend to serve the data, models and connects to the elastic database. Basically, preprocess the data and sends to the model and return results and other data.

2. front-end

    Contains all the front-end code, user interfaces etc.

3. llm_local

    Contains the Local Larger language model, in this case i am using meta llama 3 with 8 billion parameters. To start the local server for this you can run ```python test.py```


## Services Running

1. localhost:9200

    Here is running the docker container with the elastic database.

2. localhost:3000
    
    Here is running the front-end code (react)

3. localhost:5000

    Here is running the back-end flask code

4. localhost:1234

    Here is running the local large langugage Meta model


## Extra Information

### NER

- Code that runs NER
    ```tokenizer = AutoTokenizer.from_pretrained("Final_NER")```

### Load the model from the safetensors file

- Code that loads NER model
    ```model = AutoModelForTokenClassification.from_pretrained("Final_NER")```


## Techstack used
- NER (Spacy opensource finetuned model [here](https://huggingface.co/Babelscape/cner-base))
- Machine Learning (pytorch)
- Frontend - React, Typescript,
- Backend - Flask
- Local LLM - Llama.cpp
- (Optional) Electron App


## Machine Learning Model Process

1. Import Libraries:

    The necessary libraries for data handling, machine learning, and text processing are imported. This includes libraries like numpy and pandas for data manipulation, sklearn and imblearn for machine learning tasks, nltk for natural language processing, and joblib for saving and loading models.

2. Download NLTK Resources:

    The code downloads essential NLTK resources such as tokenizers, stopwords, and WordNet, which are necessary for text preprocessing tasks.

3. Load NLTK Stopwords and Initialize Porter Stemmer:

    English stopwords are loaded into a set for easy reference, and the Porter stemmer is initialized for stemming words during text preprocessing.

4. Define Preprocess Text Function:

    A function preprocess_text is defined, which tokenizes the input text, converts it to lowercase, stems each word, and removes any stopwords and non-alphabetic tokens. The processed tokens are then joined back into a single string.

5. Load Training Data:

    The dataset is loaded from a CSV file. Several columns of text data are combined into a single column named combined_text. This combined text is then processed using the preprocess_text function to generate a cleaned version called processed_text.

6. Class Distribution Analysis and Manual Oversampling:

    The distribution of classes in the target variable is analyzed. Classes with fewer samples than a specified threshold are manually oversampled to ensure a minimum number of samples per class, which helps in balancing the dataset.

7. Vectorize Text Using TF-IDF:

    The processed text data is converted into numerical features using the TF-IDF vectorizer. This step transforms the text data into a format suitable for machine learning algorithms.

8. Encode Target Variable:

    The target class labels are encoded into numerical format using a label encoder. This step is necessary for the machine learning algorithms to process the target variable.

9. Apply SMOTE:

    SMOTE (Synthetic Minority Over-sampling Technique) is applied to the dataset to generate synthetic samples for the minority classes, further balancing the dataset.

10. Split Data into Train and Test Sets:

    The dataset is split into training and testing sets using a standard train-test split. This allows for evaluating the model's performance on unseen data.

11. Define and Train Multiple Classifiers:

    Three different classifiers are defined: a Random Forest classifier, an XGBoost classifier, and a Logistic Regression classifier.

12. Create an Ensemble of Classifiers:

    An ensemble classifier is created using a voting classifier. This ensemble combines the predictions of the three individual classifiers using soft voting, which means it averages the predicted probabilities from each classifier.


13. Train the Ensemble:

    The ensemble classifier is trained on the training data. This step involves fitting the model to the data to learn the patterns and relationships between the features and the target variable.


14. Save the Trained Model and Components:

    The trained ensemble model, TF-IDF vectorizer, and label encoder are saved to disk using joblib. This allows for easy loading and reuse of the model and its components without retraining.


15. Evaluate the Model:

    The trained ensemble model is used to make predictions on the test set. The accuracy of these predictions is calculated and printed. Additionally, a detailed classification report is generated, providing precision, recall, and F1-score for each class, which helps in understanding the model's performance across different categories.



## Elastic Database

- Follow the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) to download the elastic database image from docker and then generate a container for it.

- Then follow the following python code to add you data in.
    ```` 
    from datetime import datetime
    from elasticsearch import Elasticsearch

    client = Elasticsearch(['https://localhost:9200'], basic_auth=('elastic', '***'),  verify_certs=False)

    email_templates = [
    {
        'subject': '',
        'body': '\',
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }]

    for i, template in enumerate(email_templates, start=1):
        res = client.index(index='email_templates', id=i, document=template)
        print(f"Indexed template {i}: {res['result']}")

    ````

- This is a template code to add data to your elastic database on docker.




