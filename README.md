# Article Ranking with ChatGPT extension

## Getting and preparing the Data

## ChatGPT integration
Do this in your environment:
To make the ChatGPT integration work (Mac and Linux):
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Using venv (Mac and Linux)
To create virtual environment the first time use:
```bash
python -m venv env
```

Activate env on windows:
```bash
.\env\Scripts\activate
```

Activate on Mac/Linux:
```bash
source env/bin/activate
```

To deactivate env:
```bash
deactivate
```

To install the requirements.txt run:
```bash
pip install -r requirements.txt
```


# Rank Articles with BM25

This project ranks articles using the BM25 algorithm. It provides functionality for preprocessing raw data and performing search queries on the processed dataset.

## Prerequisites

### ChatGPT Integration
To make the ChatGPT integration work:

#### Mac and Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

#### Windows
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

### Using venv

#### Create a Virtual Environment
To create a virtual environment for the first time:

```bash
python -m venv env
```

#### Activate the Virtual Environment

#### Mac and Linux
  ```bash
  source env/bin/activate
  ```

#### Windows
  ```powershell
  .\env\Scripts\activate
  ```

#### Deactivate the Virtual Environment
```bash
deactivate
```

#### Install Dependencies
To install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the Data (Optional)

Preprocessing converts raw data into a format optimized for BM25 ranking. **This step is typically not needed because the preprocessed file is already included in the repository.**

If preprocessing is required, run the following command:

```bash
python rank_articles.py preprocess
```

This command:
- Reads the raw dataset (`data/dataset.jsonl`)
- Applies preprocessing steps
- Saves the preprocessed data to `data/preprocessed_dataset.jsonl`

### 2. Search Articles Using BM25

To search for articles based on a query, use the following command:

```bash
python rank_articles.py search
```

You will be prompted to input a search term. The script will rank articles based on the BM25 algorithm and display the top results.


# Notes and TODOs
Search query that works for testing:
```
nike air jordan
```

## TODOs:
- Idealy we preprocess the intial dataset and remove all the chinese, japanese, korean characters so we dont have mistakes when plotting 
- I think we should also prepocess the titles and add them to the body that we search so that we also search the article title
- Other than that I believe our system works

NOW: we need to do eval and documentation and so on ....