# 🏦 Banking App Reviews — UK Market

This project analyzes **user reviews of UK banking apps** collected from the **Google Play Store**, with the goal of uncovering insights about customer satisfaction, strengths, and areas for improvement across different banks.

---

## 📊 Project Overview

The analysis focuses on:
- **Average user ratings** and their **evolution over time**
- **Volume of reviews** per app
- **Key topics** discussed by users in positive and negative reviews
- **Comparison between apps** in terms of user sentiment and main concerns

To achieve this, I combined **text analytics**, **topic modeling**, and **interactive visualization** techniques to better understand what drives user perceptions of digital banking apps in the UK.

---

## 🔍 Data Source

- **Source:** Google Play Store reviews  
- **Scope:** Main UK banking apps: Barclays, HSBC, Lloyds, Monzo, Revolut and Santander.  
- **Collection method:** Scraping using google_play_scraper.
- **Collected fields:** app_name, score, review_text, review_date, thumbs_up, reply, reply_date and app_version.

---

## 🧠 Methodology

1. **Data Collection**  
   Scraped reviews from the Google Play Store using google_play_scraper through `reviews_core/scraper.py`.

2. **Preprocessing & Exploratory Analysis**  
   - Exploratory analysis of the several variables, including rating distributions and trends.  
   - Cleaning and normalization of text using `reviews_core/cleaning.py`.

3. **Sentiment Analysis**
   VADER sentiment analysis performed to identify the prevailing sentiment (positive, negative or neutral) in the reviews. 
   As expected, sentiment analysis proved to be significantly redundant with the already existent 'score' information (i.e., rating given by the user).

4. **Topic Modeling**  
   - **LDA (Latent Dirichlet Allocation)** — implemented in Notebook `4.1. LDA Modelling.ipynb`  
   - **BERTopic** — implemented in Notebook `4.2. BERTopic Modelling.ipynb`  
   The BERTopic model achieved better performance, providing more **coherent and interpretable clusters**.

4. **Visualization & Deployment**  
   - Interactive dashboard built with **Streamlit** (`app.py`)  
   - Aggregated metrics and charts for rating trends  
   - Key topics and subtopics visualization for both positive and negative reviews
   - Possibility to deep dive on the reasons beneath each topic/subtopic with word cloud (`reviews_core/word_cloud.py`) and search reviews (`reviews_core/get_sample.py`) features.
---

## 🧾 Project Structure

```text
assets/
├── dfs_pipeline/              # Where intermediate datasets are stored
├── models/                    # Where topic models are stored
├── df_monthly.parquet         # Monthly aggregated app statistics
├── df_topic.parquet           # Final topic modeling results
│
notebooks/                     # Notebooks with detailed analysis and explanation of model training
├── 1. Data Collection.ipynb
├── 2. Preprocessing and EDA.ipynb
├── 3. Sentiment Analysis.ipynb
├── 4.1. LDA Modelling.ipynb
└── 4.2. BERTopic Modelling.ipynb
│
reviews_core/                  # Custom Python scripts for scraping & processing
├── __init__.py
├── scraper.py                 # Scrape new reviews and consolidate with existing data
├── cleaning.py                # Cleans new reviews and consolidates with existent data
├── apply_bertopic.py          # Applies bertopic model (already trained) on new reviews
├── update_final_dataframes.py # Updates final dataframes that are used in production
├── word_cloud.py              # Create a word cloud for selected filters
├── get_sample.py              # Return a sample of reviews given user filters
│
app.py                         # Streamlit dashboard
requirements.txt               # Python dependencies
runtime.txt                    # Streamlit runtime version
run_pipeline.py                # Automatic pipeline for data updates
README.md                      # Project description
```

## 🚀 How to Run

1. **Clone the repository**
   git clone https://github.com/yourusername/banking_app_reviews.git
   cd banking_app_reviews

2. **Install dependencies**
   pip install -r requirements.txt

3. **Run the Streamlite dashboard**
   streamlit run app.py

4. **Run run_pipeline.py** to update with new reviews
