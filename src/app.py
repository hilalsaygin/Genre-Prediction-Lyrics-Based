import streamlit as st
import pandas as pd
import re
from joblib import load
from scipy.sparse import vstack
from sklearn.metrics import classification_report, accuracy_score


def split_user_input(lyrics, tag="Unknown"):
    chorus_data = []
    non_chorus = []
    sections = re.split(r'\[(.*?)\]', lyrics)

    if len(sections) == 1:  # No sections found; treat all as chorus
        chorus_data.append({"tag": tag, "lyrics": lyrics.strip()})
    else:
        if sections[0].strip():
            content = sections[0].strip()
            chorus_data.append({"tag": tag, "lyrics": content})

        for i in range(2, len(sections), 2):
            if sections[i - 1] == '' or sections[i] == '':
                continue
            section_type = sections[i - 1]
            content = sections[i].strip()
            if content:
                data = {"tag": tag, "lyrics": content}
                if 'Chorus' in section_type:
                    chorus_data.append(data)
                else:
                    non_chorus.append(data)

    # If no chorus was found, consider all lyrics as a single chorus
    if not chorus_data:
        chorus_data.append({"tag": tag, "lyrics": lyrics.strip()})

    chorus_df = pd.DataFrame(chorus_data)
    non_chorus_df = pd.DataFrame(non_chorus)
    return chorus_df, non_chorus_df


def preprocess_user_input(lyrics, tfidf, svd, weight1=1.5, weight2=1.0):
    chorus_df, non_chorus_df = split_user_input(lyrics)

    # Extract lyrics from DataFrames
    chorus_text = " ".join(chorus_df['lyrics']) if not chorus_df.empty else ""
    non_chorus_text = " ".join(non_chorus_df['lyrics']) if not non_chorus_df.empty else ""

    # Transform chorus and non-chorus parts
    chorus_features = tfidf.transform([chorus_text]) * weight1
    non_chorus_features = tfidf.transform([non_chorus_text]) * weight2

    # Combine the weighted features
    combined_features = vstack([chorus_features, non_chorus_features])
    if svd:
        # Apply dimensionality reduction (SVD)
        reduced_features = svd.transform(combined_features)

        return reduced_features
    return combined_features

def test_model(dump_path):
    # Load the saved components
    model_data = load(dump_path)
    tfidf = model_data['tfidf']
    svd = model_data['svd']
    classifier = model_data['pipeline'].named_steps['classifier']

    st.title("Genre Predictor")

    # Single prediction
    st.header("1. Predict Genre for a Single Lyrics Input")
    user_lyrics = st.text_area("Enter your song lyrics:", height=150)

    if st.button("Predict Genre"):
        if user_lyrics.strip():
            # Preprocess the input lyrics
            features = preprocess_user_input(user_lyrics, tfidf, svd)
            prediction = classifier.predict(features)[0]
            st.subheader(f"**Predicted Genre:** {prediction}")
        else:
            st.warning("Please enter lyrics before clicking Predict Genre.")

    # Batch evaluation
    st.header("2. Evaluate on a Test Set (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file with two columns: 'lyrics' and 'tag'", type=['csv'])

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)

            if 'lyrics' in test_df.columns and 'tag' in test_df.columns:
                y_true = test_df['tag']
                predictions = []

                for lyrics in test_df['lyrics']:
                    try:
                        features = preprocess_user_input(lyrics, tfidf, svd)
                        predictions.append(classifier.predict(features)[0])
                    except Exception as e:
                        predictions.append("Error")  # Handle problematic rows

                acc = accuracy_score(y_true, predictions)
                report = classification_report(y_true, predictions, zero_division=0)
                st.write("**Accuracy:**", round(acc * 100, 2), "%")
                st.text("Classification Report:")
                st.text(report)
            else:
                st.error("The uploaded file must contain 'lyrics' and 'tag' columns.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "genre_prediction_model"
    test_model(model_path + '.joblib')
