import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"  # FastAPI backend

# Page configuration
st.set_page_config(
    page_title="AI Traveling Agent", 
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (same as your version, skipped here for brevity) ---
# Keep your big CSS block exactly as is

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧳 AI Traveling Agent</div>
    <div class="hero-subtitle">Your intelligent companion for all travel-related questions and planning</div>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("""
<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon">🗺️</div>
        <div class="feature-title">Trip Planning</div>
        <div class="feature-desc">Get personalized itineraries and destination recommendations</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💰</div>
        <div class="feature-title">Budget Advice</div>
        <div class="feature-desc">Find the best deals and budget-friendly travel options</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🏨</div>
        <div class="feature-title">Accommodation</div>
        <div class="feature-desc">Discover perfect places to stay for your journey</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🍽️</div>
        <div class="feature-title">Local Cuisine</div>
        <div class="feature-desc">Explore authentic local food and dining experiences</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("### 💭 Ask me anything about travel!")

    # Example quick-questions
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("🏖️ Best beaches in Bali", key="beach"):
            st.session_state.question = "What are the best beaches to visit in Bali?"
        if st.button("🗼 Paris 3-day itinerary", key="paris"):
            st.session_state.question = "Can you suggest a 3-day itinerary for Paris?"
    with example_col2:
        if st.button("💵 Budget travel tips", key="budget"):
            st.session_state.question = "What are some budget travel tips for backpackers?"
        if st.button("📋 Visa requirements", key="visa"):
            st.session_state.question = "What are the visa requirements for traveling to Japan?"

    # Main input
    question = st.text_input(
        "✍️ Type your travel question here:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., What's the best time to visit Tokyo?",
        key="main_input"
    )

    # Center the ask button
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        ask_button = st.button("🚀 Ask Agent", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Handle Question
if ask_button and question.strip():
    with st.spinner("🤖 AI Agent is thinking..."):
        try:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(1)  # UX delay

            # ✅ FIX: send "query" instead of "question"
            response = requests.post(f"{API_URL}/ask", json={"query": question})
            loading_placeholder.empty()

            if response.status_code == 200:
                data = response.json()
                answer_text = data.get("answer", "No answer returned.")

                st.markdown(f"""
                <div class="answer-container">
                    <div class="answer-title">🎯 Your Travel Answer</div>
                    <div class="answer-text">{answer_text}</div>
                </div>
                """, unsafe_allow_html=True)

                if 'question' in st.session_state:
                    del st.session_state.question
            else:
                st.error(f"🚫 Backend error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Error: Make sure FastAPI server is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

elif ask_button and not question.strip():
    st.warning("⚠️ Please enter a travel question to get started!")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
    Made with ❤️ for travelers around the world | Powered by AI
</div>
""", unsafe_allow_html=True)
