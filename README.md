# Food Review Sentiment Analyzer

A web-based application that analyzes the sentiment of food reviews using Machine Learning. Built with Streamlit and Linear SVC, this tool helps determine whether a food review expresses positive, negative, or neutral sentiment.

## 🌟 Features

- Real-time sentiment analysis of food reviews
- User-friendly web interface
- Pre-defined sample reviews for testing
- Text preprocessing and cleaning
- Support for positive, negative, and neutral sentiment classification
- Interactive results with emoji indicators

## 📊 Technical Details

- **Model**: Linear Support Vector Classification (LinearSVC)
- **Accuracy**: ~73% on the training dataset
- **Text Processing**: NLTK for lemmatization and stopword removal
- **Vectorization**: TF-IDF Vectorizer
- **Dataset**: Trained on approximately 100K food reviews

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Reviews-Sentiment-Analyzer
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main Streamlit application file
- `utils.py`: Text preprocessing utilities
- `Cleaned_Reviews.csv`: Preprocessed dataset
- `Reviews.csv`: Original dataset
- `Sentiment_Analysis_Prototype.ipynb`: Jupyter notebook containing model development
- `best_model.pkl`: Trained LinearSVC model (not included in repo)
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer (not included in repo)

## 🛠️ Usage

1. Launch the application using `streamlit run app.py`
2. Enter a food review in the text area or use one of the sample reviews
3. Click "Analyze Sentiment" to get the result
4. The result will be displayed with an appropriate emoji indicator:
   - ✅ Positive: Review expresses satisfaction
   - ❌ Negative: Review indicates dissatisfaction
   - 🟡 Neutral: Review is mixed or indifferent

## ⚠️ Limitations

- Model is trained on a sample of ~100K reviews
- Best performance on clear positive/negative reviews
- Current accuracy is around 73% (could be improved with full dataset)
- Limited to English language reviews

## 👨‍💻 Author

- **Sushan Bajracharya**

## 🙏 Acknowledgments

- Dataset based on food reviews collection
- Built using Streamlit framework
- NLTK for natural language processing
- scikit-learn for machine learning implementation