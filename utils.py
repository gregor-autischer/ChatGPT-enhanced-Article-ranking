# utils.py

# functions to generate HTML and plots and to check API Key are here to declutter the main script

import os
import matplotlib.pyplot as plt
import numpy as np

def check_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"OPENAI_API_KEY is set: {api_key[:5]}...")  # Print the first few characters for confirmation
    else:
        print("OPENAI_API_KEY is not set.")

def generate_html(top_results_simple, top_results_expanded, output_file, html_folder):
    # Create a basic HTML structure
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Top Articles</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                padding: 20px;
                background-color: #f9f9f9;
                color: #333;
            }}
            h1 {{
                text-align: center;
                color: #444;
            }}
            .container {{
                display: flex;
                gap: 20px;
            }}
            .column {{
                flex: 1;
                padding: 10px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow-y: auto;
            }}
            .column h2 {{
                text-align: center;
                color: #007BFF;
            }}
            .article {{
                margin-bottom: 20px;
                padding: 15px;
                border-bottom: 1px solid #ddd;
            }}
            .article:last-child {{
                border-bottom: none;
            }}
            .article h3 {{
                margin-top: 0;
                color: #007BFF;
            }}
            .article p {{
                margin: 10px 0 0;
            }}
        </style>
    </head>
    <body>
        <h1>Top Articles</h1>
        <div class="container">
            <div class="column">
                <h2>Simple Search Results</h2>
                {simple_articles}
            </div>
            <div class="column">
                <h2>Expanded Search Results</h2>
                {expanded_articles}
            </div>
        </div>
    </body>
    </html>
    """

    # Generate HTML content for simple search articles
    simple_articles_html = ""
    for _, row in top_results_simple.iterrows():
        title = row.get("title_abc", "No Title")
        content = row.get("content_abc", "No Content")
        simple_articles_html += f"""
        <div class="article">
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
        """

    # Generate HTML content for expanded search articles
    expanded_articles_html = ""
    for _, row in top_results_expanded.iterrows():
        title = row.get("title_abc", "No Title")
        content = row.get("content_abc", "No Content")
        expanded_articles_html += f"""
        <div class="article">
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
        """

    # Fill the HTML template with the articles
    full_html = html_content.format(
        simple_articles=simple_articles_html,
        expanded_articles=expanded_articles_html
    )

    os.makedirs(html_folder, exist_ok=True)  # Create the folder if it doesn't exist
    html_file_path = os.path.join(html_folder, output_file)

    # Save the HTML to a file
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"HTML file with top articles saved as {html_file_path}")

def create_plot(top_results, score_column, query_type, plot_file_path):
    # Truncate titles for better display
    top_results['truncated_title'] = top_results['title_abc'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x
    )

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.barh(top_results['truncated_title'], top_results[score_column], color='skyblue')
    plt.xlabel('BM25 Score', fontsize=12)
    plt.ylabel('Articles', fontsize=12)
    plt.title(f"Top Ranked Documents ({query_type})", fontsize=14)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(plot_file_path)  # Save the plot to the specified file
    plt.close()  # Close the plot to free resources

    print(f"Plot saved to {plot_file_path}")

def plot_line_comparison(y_true, y_pred, query_type, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Ground Truth", marker='o', linestyle='-', alpha=0.8)
    plt.plot(y_pred, label="BM25 Predicted Scores", marker='x', linestyle='--', alpha=0.8)
    plt.xlabel("Document Index")
    plt.ylabel("Relevance Score")
    plt.title(f"Ground Truth vs Predicted Relevance ({query_type})")
    plt.legend()
    plt.grid(alpha=0.4)

    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

def plot_absolute_error_by_index(y_true, y_pred, query_type, plot_path):
    absolute_errors = np.abs(y_pred - y_true)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(absolute_errors)), absolute_errors, color='orange', alpha=0.7)
    plt.xlabel("Document Index")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Errors by Document ({query_type})")
    plt.grid(alpha=0.4)

    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")