# Rank Articles with BM25 and ChatGPT extension

This project ranks articles using the BM25 algorithm. It provides functionality for preprocessing raw data and performing search queries on the processed dataset. This project integrates the BM25 ranking algorithm with ChatGPT to improve information retrieval. By leveraging ChatGPT's advanced language processing capabilities, user queries are refined to better capture intent, resulting in more accurate BM25 rankings. The approach focuses on efficiency by using short and targeted prompts, keeping computational demands low. This combination creates a practical and effective solution for search tasks, balancing simplicity with advanced contextual understanding.

## Prerequisites

### ChatGPT Integration
To make the ChatGPT integration work:
ATTENTION: replace >your_api_key_here< with the api key I sent in discord
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

You will be prompted to input a search term. The script will rank articles based on the BM25 algorithm and also with a ChatGPT enhanced query and display the top results.

## Where is What?
Once you run the search, the script will automatically create several files as output. 

### Ranked Articels
The ranked articels can be found inside the html folder. Open the created top_results.html in your browser to see the ranked articels for both the standard query and the expanded query side by side.

### Plots
Several plots are also created. Inside the output_plots folder several plots are created. These plots show the absolute error and precision and recall for both the normal and the expanded query. Also there are plots that show the 10 best results with their respective BM25 scores for both the normal and the expanded query.

### Data
All the necessary data can be found in the data folder. The file dataset.jsonl is our original dataset that we used. This code when preprocessing the data then creates the preprocessed_dataset.jsonl file, which represents the preprocessed data that is then subsequently used in the code.