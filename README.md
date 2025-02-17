# **Lyrics-Based Genre Classification: Structure-Aware Machine Learning Approach**  

## **Table of Contents**  
1. [Introduction](#introduction) 
2. [Dataset Construction](#dataset-construction)  
3. [Feature Extraction](#feature-extraction)  
4. [Experimental Setup](#experimental-setup)  
5. [Results & Contributions](#results--contributions)  
6. [Installation & Environment Setup](#installation--environment-setup)  
7. [Usage](#usage)  
8. [Future Work](#future-work)  

---  

## **Introduction**  
Focuses on **lyrics-based genre classification**, leveraging both **textual and structural** properties of song lyrics to enhance classification accuracy. Unlike traditional methods that treat lyrics as a uniform text block, this approach prioritizes **song composition**, particularly **chorus sections**, to evaluate their impact on prediction performance.  

A **novel dataset** was created using **Genius.com** and **HuggingFace**, incorporating **section-level labels** such as chorus, verse, and intro. The dataset was then segmented into **chorus and non-chorus subsets** to enable comparative analysis of their contributions to classification accuracy.  

A **custom feature extraction system** was developed, integrating **textual, semantic, and stylistic features** such as:  
✔ **TF-IDF, word embeddings, and n-grams**  
✔ **Rhyme patterns and sentiment analysis**  
✔ **Grammatical properties and structural metrics**  

The extracted feature sets were transformed into **sparse matrix representations** and evaluated using **Random Forest, Multinomial Naïve Bayes, and Logistic Regression models**. Experiments were conducted by training models on:  
- **Chorus-only lyrics**  
- **Non-chorus lyrics**  
- **Combined data with weighted contributions**  

The results demonstrated that **prioritizing chorus features significantly improved classification accuracy**, with the **best-performing configuration achieving ~60-61% accuracy** using **Random Forest**. This work highlights the importance of **structural elements in lyrics classification** and provides a **scalable framework** for integrating structural properties into text-based classification tasks.  

**Keywords:** Machine Learning, Lyrics-Based Genre Detection, Feature Engineering, Random Forest  

---  

The key objectives of this research are:  
- **Investigate the role of chorus structures** in genre prediction  
- **Develop a dataset** that segments lyrics into structured sections  
- **Implement feature engineering techniques** to extract meaningful patterns  
- **Evaluate multiple classification models** and compare their performances  

By addressing these goals, this study contributes to **music information retrieval, recommendation engines, and text classification applications** beyond the music industry.  

---  

## **Dataset Construction**  
The dataset was built by combining lyrics from:  
- **Genius.com** (crowdsourced lyrics with section labels)  
- **HuggingFace** (lyric datasets for NLP research)  

Each song was **annotated with structural labels** (e.g., `[Chorus]`, `[Verse]`, `[Intro]`), allowing segmentation into distinct subsets:  
✔ **Chorus** lyrics  
✔ **Non-chorus** lyrics  
✔ **Full lyrics (combined & weighted for impact analysis)**  

To ensure **genre balance**, the dataset was curated to **contain a uniform distribution** of genres, preventing model bias.  

---  

## **Feature Extraction**  
A **custom feature extraction pipeline** was developed to capture different aspects of lyrics:  

### **1️⃣ Statistical Features**  
- **Word frequency, lexical diversity, punctuation ratios**  

### **2️⃣ Semantic Features**  
- **Word embeddings** using **Word2Vec**  
- **TF-IDF & Bag-of-Words** models  

### **3️⃣ Structural Features**  
- **Rhyme patterns, sentiment scores, stylistic metrics**  
- **Chorus vs. non-chorus weighting** for comparative analysis  

The extracted features were transformed into **sparse matrices** for machine learning models.  

---  

## **Experimental Setup**  
To analyze the impact of **chorus prioritization**, multiple training configurations were tested:  

✔ **Model trained on chorus data only**  
✔ **Model trained on non-chorus data only**  
✔ **Model trained on combined data (with different weightings for chorus and non-chorus features)**  
✔ **Feature engineering variations (TF-IDF vs. embeddings, rhyme patterns, sentiment analysis, etc.)**  
✔ **Dimensionality reduction using SVD (Singular Value Decomposition)**  

The following **machine learning models** were used:  
- **Random Forest**  
- **Multinomial Naïve Bayes**  
- **Logistic Regression**  

Metrics for evaluation:  
✔ **Accuracy**  
✔ **Precision, Recall, F1-score**  
✔ **Comparison of feature contributions (chorus vs. non-chorus)**  

---  

## **Results & Contributions**  
📌 **Key Finding:** **Prioritizing chorus features improves classification accuracy**.  

💡 **Best Configuration:**  
- Using **both chorus and non-chorus lyrics** with **weighted chorus emphasis**  
- **Random Forest achieved ~60-62% accuracy**, outperforming baseline models  

🎯 **Contributions of this study:**  
✔ **Introduced a structure-aware approach** to lyrics classification  
✔ **Demonstrated that chorus sections carry strong genre signals**  
✔ **Provided a scalable feature extraction and weighting system**  
✔ **Showed how structural elements improve text-based classification**  

---  

## **Installation & Environment Setup**  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/hilalsaygin/Genre-Prediction-Lyrics-Based.git
cd Genre-Prediction-Lyrics-Based
```

2️⃣ **Create a Virtual Environment**  
```bash
python -m venv venv  
source venv/bin/activate  # For macOS/Linux  
venv\Scripts\activate  # For Windows
```

3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

4️⃣ **Run Jupyter Notebook**  
```bash
jupyter notebook
```
Then navigate to **`src/process_lyrics.ipynb`** and execute the notebook.

---  

## **Future Work**  
🔹 **Explore deep learning approaches (e.g., LSTMs, transformers)**  
🔹 **Enhance dataset with more genre-balanced samples**  
🔹 **Incorporate additional linguistic and phonetic features**  

---  

## **Conclusion**  
This project introduces a **structure-aware approach** to lyrics-based genre classification, proving that **chorus prioritization enhances accuracy**. By integrating **custom feature extraction techniques**, **sparse matrix representations**, and **machine learning models**, the study provides a **scalable framework** for **text-based classification** beyond music genres.

🚀 **Potential Applications:**  
✅ Music recommendation systems  
✅ Automated genre tagging for streaming platforms  
✅ Enhanced NLP-based text classification  

---  

## **License**  
This project is open-source and available under the **MIT License**.

🔥 **If you find this project useful, give it a ⭐ on GitHub!** 🔥  
