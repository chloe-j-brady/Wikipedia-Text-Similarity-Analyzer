#!/usr/bin/env python3
#Chloe Brady 
import requests
from bs4 import BeautifulSoup
import nltk
import ssl
import os
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# fix the error with nltk downloading on a Mac, as Python often fails
try:
	_create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
	pass
else:
	ssl._create_default_https_context = _create_unverified_https_context
	
# Download NLTK data 
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

#Chicken website to scrape data from
url = "https://en.wikipedia.org/wiki/Chicken"

headers = {
	"User-Agent": "Mozilla/5.0"
}
#request the main chicken Wikipedia page 
response = requests.get(url, headers=headers)
#Parse HTML using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')


container = soup.find(id="mw-content-text")
# this list will store the real page titles for Part 3
page_titles = []
#Extract only the paragraph text first from the main article
if container:
	content_div = container.find(class_="mw-parser-output")
	
	if content_div:
		# Extracts the main plain text content from the page
		text = ""
		for p in content_div.find_all("p"):
			text += p.get_text() + "\n"
			
		# saving the 1st chicken page
		with open("Chicken.txt", "w", encoding="utf-8") as f:
			f.write(text)
			
		# save the real title of the first page
		if soup.title:
			page_titles.append(soup.title.get_text())
		else:
			page_titles.append("Chicken")
			
		# get first 20 valid article links from article text only
		links = []
		for p in content_div.find_all("p"):
			for link in p.find_all("a", href=True):
				url = link.get("href")
				
				# URL common pitfalls, Relative URLs:
				# used to help me understand how to use with a correct if statement for valid article links 
				if (url and url.startswith("/wiki/")
					and ":" not in url          # remove help, file, special pages
					and "#" not in url          # remove section links
					and "disambiguation" not in url.lower()  # remove disambiguation pages
					):
					full_url = f"https://en.wikipedia.org{url}"
					
					# add link to the list if it is not already in the list
					if full_url not in links:
						links.append(full_url)
						
				# get first 20 links
				if len(links) == 20:
					break
			if len(links) == 20:
				break
			
		# Now visit each of the 20 linked pages
		for i, link in enumerate(links):
			# Send request to each Wikipedia page
			page = requests.get(link, headers=headers)
			# Parse the HTML content
			soup2 = BeautifulSoup(page.text, "html.parser")
			
			# save the title for this linked page
			if soup2.title:
				page_titles.append(soup2.title.get_text())
			else:
				page_titles.append(f"Page {i+1}")
				
			# find the main content section of the page (same structure as original page)/ same as code above 
			subContainer = soup2.find(id="mw-content-text")
			# find the paragraph content inside the main section
			if subContainer:
				subContent = subContainer.find(class_="mw-parser-output")
			else:#prevent crashing 
				subContent = None
			#empty string each time for the for loop
			text2 = ""
			#get all the text and combine into one string
			if subContent:
				#gets all <p> tags - paragraphs
				for p in subContent.find_all("p"):
					#get text only, remove extra spaces, put new paragraph on a new line
					text2 += p.get_text().strip() + "\n"
					
			# only save if text exists - prevents empty files
			if text2.strip():
				#save the cleaned article
				with open(f"page_{i+1}.txt", "w", encoding="utf-8") as f:
					f.write(text2)
					
# Part 2:

					
# Create stop word set and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(raw_text):
	
	# 1: Conversion to lowercase
	lower_text = raw_text.lower()
	
	# 2: Tokenization using the library
	# splits text into individual words
	tokens = word_tokenize(lower_text)
	
	# 3: Stop word elimination
	# removes common words and anything that is not a word (punctuation, numbers, etc.)
	filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
	
	# 4: Lemmatization
	# convert each word to its base form
	lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
	
	return lemmatized_tokens

# List of all 21 files from part 1
file_list = ["Chicken.txt"] + [f"page_{i}.txt" for i in range(1, 21)]
		
#used count later that all 21 files have been processed 
processed_count = 0
# loop through every file
for filename in file_list:
	if os.path.exists(filename):
		# open and read the file
		with open(filename, "r", encoding="utf-8") as f:
			raw_text = f.read()
		# apply the preprocessing function from above to the raw wiki text
		processed_tokens = preprocess_text(raw_text)
		# Keep a clean processed version of each document
		# Not overwriting the original raw file
		output_name = "processed_" + filename
		# open "clean" file and join tokens together
		with open(output_name, "w", encoding="utf-8") as f:
			f.write(" ".join(processed_tokens))
		#increment counter
		processed_count += 1
		
	else:
		print(f"{filename} not found")
if processed_count == 21:
	print(f"Preprocessing complete! {processed_count} files processed. ")
	print("Have a clean version for all documents processed")



# Part 3: Build a Dataset
# Construct a dataset where each row corresponds to one document.
dataset = []

for i, filename in enumerate(file_list):
	if os.path.exists(filename):
		
		# read raw text
		with open(filename, "r", encoding="utf-8") as f:
			raw_text = f.read()
			
		# read processed text
		processed_filename = "processed_" + filename
		with open(processed_filename, "r", encoding="utf-8") as f:
			cleaned_text = f.read()
			
		# use the real Wikipedia title if it exists
		# if for some reason it does not, fall back to the filename
		if i < len(page_titles):
			page_title = page_titles[i]
		else:
			page_title = filename
			
		dataset.append({
			"Document_id": f"Document {i+1}",
			"Page_Title": page_title,
			"Raw_text": raw_text,
			"cleaned_text": cleaned_text
		})
		
# create dataframe
df = pd.DataFrame(dataset)

# save to csv
df.to_csv("data.csv", index=False, encoding="utf-8")

print(df.head())
print("\nDataset created!")

# Part 4: Top 5 Most Common Words

results = []

for i, row in df.iterrows():
	text = row["cleaned_text"]
	words = text.split()
	# Count word frequencies
	word_counts = Counter(words)
	# Get top 5 most common words
	top_5 = word_counts.most_common(5)
	results.append({
		"Document_id": row["Document_id"],
		"Page_Title": row["Page_Title"],
		
		# Word 1 and its frequency
		"Top_Word_1": top_5[0][0] if len(top_5) > 0 else "",
		"Frequency_1": top_5[0][1] if len(top_5) > 0 else 0,
		
		# Word 2 and its frequency
		"Top_Word_2": top_5[1][0] if len(top_5) > 1 else "",
		"Frequency_2": top_5[1][1] if len(top_5) > 1 else 0,
		
		# Word 3 and its frequency
		"Top_Word_3": top_5[2][0] if len(top_5) > 2 else "",
		"Frequency_3": top_5[2][1] if len(top_5) > 2 else 0,
		
		# Word 4 and its frequency
		"Top_Word_4": top_5[3][0] if len(top_5) > 3 else "",
		"Frequency_4": top_5[3][1] if len(top_5) > 3 else 0,
		
		# Word 5 and its frequency
		"Top_Word_5": top_5[4][0] if len(top_5) > 4 else "",
		"Frequency_5": top_5[4][1] if len(top_5) > 4 else 0
	})
	
# Create new dataframe - don't overwrite original df
common_words_df = pd.DataFrame(results)
# save results as a CSV file to create a clean table format instead of messy terminal output
common_words_df.to_csv("top5Words.csv", index=False, encoding="utf-8")
print("\nTop 5 most common words saved to top5Words.csv")


#part 5 TF-IDF
documents = df["cleaned_text"].tolist()
file_names = df ["Page_Title"].tolist()

# 1. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(documents)
# 'tfidf_matrix' contains all your scraped Wikipedia articles

# Running this once creates the entire matrix
print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

#compute cosine similarity between all documents
sim_matrix = cosine_similarity(tfidf_matrix)

#convert to a DF
sim_df = pd.DataFrame(sim_matrix, columns=file_names, index=file_names)
#print out dataframe
print("\nSimilarity Matrix:\n")
print(sim_df.round(4)) #round to 4th decimal point

#find top similar document pairs 
print("\nTop Similar Document Pairs:\n")
pairs = [] 

#loop through every article
for i in range(len(file_names)):
	
	#inner loop starts at the next item to avoid duplicates 
	for j in range(i + 1, len(file_names)):
		
		#Get the score from the similarity matrix
		score = sim_matrix[i][j]
		
		pairs.append((file_names[i], file_names[j], score))
	
	# sort from highest to lowest
pairs.sort(key=lambda x: x[2], reverse=True)

for doc1, doc2, score in pairs[:10]:
	print(f"{doc1} <--> {doc2} = {score:.3f}")
	
	
# Part 6: Visualization of cosine similarity heatmap

# Convert to DataFrame (gives axis labels automatically)
df_sim = pd.DataFrame(sim_matrix, index=file_names, columns=file_names)

# Clean up the image to make it more readable
sns.set_theme(style="white")

# Width and height of the canvas
plt.figure(figsize=(12, 10))

# Create heatmap
ax = sns.heatmap(
		df_sim, #similarity matrix
		annot=False,# don't show numbers of cosine score too messy
		cmap="coolwarm", # color
		linewidths=0.5,# grid lines
		cbar_kws={"label": "Cosine Similarity"}  #label the color bar
)
# Axes customization 
# Rotate X labels so they don't overlap
plt.xticks(rotation=45, ha='right', fontsize=10)
# Keep Y labels straight
plt.yticks(rotation=0, fontsize=10)

# Add axis titles 
plt.xlabel("Documents Being Compared", fontsize=12, fontweight='bold')
plt.ylabel("Documents", fontsize=12, fontweight='bold')

# Main title
plt.title("Cosine Similarity Between Wikipedia Articles", fontsize=16, pad=20)

# Prevent labels from getting cut off
plt.tight_layout()
#save the figure to my current folder
plt.savefig("cosine_similarity_heatmap.png", dpi=300, bbox_inches="tight")
#Always show after saving
plt.show()