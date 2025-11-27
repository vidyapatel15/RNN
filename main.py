
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time

# Set page configuration
st.set_page_config(
    page_title="IMDB Movie Review Classifier by vidya",
    page_icon="🎬
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .review-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
    }
    .positive {
        color: #00D100;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .negative {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class MovieReviewPredictor:
    def __init__(self):
        self.model = None
        self.reverse_word_index = None
        self.max_length = 500

    def load_model(self, model_path):
        """Load the trained RNN model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def load_word_index(self, word_index_path):
        """Load IMDB word index"""
        try:
            with open(word_index_path, 'rb') as f:
                self.reverse_word_index = pickle.load(f)
            return True
        except Exception as e:
            st.error(f"Error loading word index: {e}")
            return False

    def preprocess_text(self, text):
        """Convert text to sequence of integers - Following class example"""
        words = text.lower().split()

        # Convert words to integers using the word index
        sequence = []
        for word in words:
            # Clean the word
            word = word.strip('.,!?;:"()[]{}')
            # Find word in reverse_word_index
            found = False
            for idx, w in self.reverse_word_index.items():
                if w == word and idx >= 3:  # Skip reserved indices
                    sequence.append(idx)
                    found = True
                    break

        # Pad sequence - Following class example
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))

        return np.array([sequence])

    def predict_sentiment(self, text):
        """Predict sentiment of movie review"""
        sequence = self.preprocess_text(text)
        prediction = self.model.predict(sequence, verbose=0)[0][0]

        sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return sentiment, confidence, prediction

def main():
    # Title with custom styling - CHANGE YOUR LAST NAME HERE
    st.markdown('<h1 class="main-header">IMDB Movie Review Classifier by vidya</h1>',
                unsafe_allow_html=True)

    # Initialize predictor
    predictor = MovieReviewPredictor()

    # Load model and word index
    with st.spinner("Loading model and resources..."):
        model_loaded = predictor.load_model('my_movie_review_rnn.h5')
        word_index_loaded = predictor.load_word_index('imdb_word_index.pkl')

    if not model_loaded or not word_index_loaded:
        st.error("❌ Could not load required files. Please ensure model files are available.")
        return

    st.success("✅ Model and word index loaded successfully!")

    # Sample movie reviews section
    st.markdown("---")
    st.subheader("📝 Sample Movie Reviews Analysis")

    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged from beginning to end. Highly recommended!",
        "I hated this film. The plot was confusing, the characters were poorly developed, and the ending made no sense. Waste of time.",
        "A decent movie with some good moments. The cinematography was beautiful but the pacing felt slow at times. Overall, it was okay.",
        "Brilliant performance by the lead actor! The direction was innovative and the screenplay was tight. One of the best movies I've seen this year.",
        "Terrible movie. Poor acting, bad script, and awful special effects. I can't believe I sat through the entire thing."
    ]

    # Analyze sample reviews
    for i, review in enumerate(sample_reviews, 1):
        with st.container():
            st.markdown(f'<div class="review-box">', unsafe_allow_html=True)
            st.write(f"**Review {i}:**")
            st.write(f'"{review}"')

            # Predict sentiment
            sentiment, confidence, score = predictor.predict_sentiment(review)

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Sentiment**")
                if sentiment == "POSITIVE":
                    st.markdown(f'<p class="positive">{sentiment} 👍</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="negative">{sentiment} 👎</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Confidence**")
                st.write(f"**{confidence:.2%}**")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.write("**Prediction Score**")
                st.write(f"**{score:.4f}**")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # User input section
    st.markdown("---")
    st.subheader("🎬 Analyze Your Own Review")

    user_review = st.text_area(
        "Enter a movie review to analyze:",
        height=150,
        placeholder="Type your movie review here...\nExample: 'This movie was amazing with great acting and an engaging plot!'"
    )

    if st.button("🔍 Analyze Sentiment", type="primary") and user_review:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(1)  # Simulate processing time
            sentiment, confidence, score = predictor.predict_sentiment(user_review)

            st.markdown("---")
            st.subheader("📊 Analysis Results")

            # Results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", sentiment, delta="Positive" if sentiment == "POSITIVE" else "Negative")

            with col2:
                st.metric("Confidence", f"{confidence:.2%}")

            with col3:
                st.metric("Raw Score", f"{score:.4f}")

            # Visual feedback
            if sentiment == "POSITIVE":
                st.success(f"🎉 This review is **POSITIVE** with {confidence:.2%} confidence!")
            else:
                st.error(f"👎 This review is **NEGATIVE** with {confidence:.2%} confidence!")

            # Confidence progress bar
            st.write("**Model Confidence:**")
            st.progress(float(confidence))

if __name__ == "__main__":
    main()
