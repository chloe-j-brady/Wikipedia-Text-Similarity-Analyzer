# Wikipedia Text Similarity Analyzer

This project is a Python-based text mining pipeline that collects Wikipedia article text, cleans and preprocesses the content, analyzes word frequency, and measures document similarity using TF-IDF and cosine similarity.

The project starts with the Wikipedia page for "Chicken" and gathers text from that page plus 20 related Wikipedia articles linked from the main article. Each article is treated as a separate document, allowing the program to compare how similar the articles are based on their text content.

## Project Goals

The goal of this project was to practice the full data science workflow using unstructured text data:

- Collect text data from web pages
- Clean and preprocess natural language text
- Build a structured dataset from raw web content
- Analyze the most common words in each document
- Convert text into numerical features using TF-IDF
- Compare document similarity using cosine similarity
- Visualize similarity patterns with a heatmap

## Tools and Libraries

- Python
- requests
- BeautifulSoup
- pandas
- NLTK
- scikit-learn
- matplotlib
- seaborn

## How It Works

1. **Web Scraping**  
   The program starts from the Wikipedia page for "Chicken" and extracts the main article text. It then collects the first 20 valid article links from the page and scrapes text from each linked article.

2. **Text Preprocessing**  
   Each document is cleaned by converting text to lowercase, tokenizing words, removing stop words, and applying lemmatization.

3. **Dataset Creation**  
   The cleaned and raw text are stored in a structured CSV file, with each row representing one Wikipedia article.

4. **Word Frequency Analysis**  
   The program identifies the five most common words in each processed document and saves the results to a separate CSV file.

5. **TF-IDF and Cosine Similarity**  
   The cleaned documents are transformed into TF-IDF vectors. Cosine similarity is then used to compare how similar each document is to every other document.

6. **Visualization**  
   A heatmap is generated to show the similarity scores between the 21 Wikipedia articles.

## Outputs

- `data/data.csv` — structured dataset containing raw and cleaned article text
- `data/top5Words.csv` — top five most common words for each article
- `visualizations/similarity_heatmap.png` — heatmap showing cosine similarity between documents

## What I Learned

This project helped me understand how raw text can be transformed into usable data for analysis. I practiced scraping web data, cleaning natural language text, creating TF-IDF features, and interpreting similarity scores. One of the biggest challenges was making sure the scraper only collected relevant article links and ignored navigation bars, citations, images, and special Wikipedia pages.

## Future Improvements

In the future, I would like to expand this project by:

- Comparing stemming and lemmatization results
- Adding more advanced NLP techniques
- Creating a network graph of article similarity
- Allowing the user to enter any starting Wikipedia page
- Building an interactive dashboard for exploring similarity results