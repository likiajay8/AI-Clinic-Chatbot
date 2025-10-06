import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import string

# ---------------------
# Ensure NLTK resources are downloaded
# ---------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# ---------------------
# Clinic data and training phrases
# ---------------------
clinic_data = {
    "timing": "Our clinic is open from 9 AM to 8 PM, Monday to Saturday.",
    "sunday": "We are closed on Sundays.",
    "doctor_availability": "Dr. Mehta is available from 10 AM to 1 PM, Dr. Sharma from 4 PM to 8 PM.",
    "fees": "Our consultation fees are ₹500 per visit.",
    "contact": "You can call us at 9876543210 for appointments.",
    "location": "We are located at MG Road, Bengaluru. Location link: https://maps.app.goo.gl/jt3suvjK5yxnvdXd7",
    "appointment": "Call us on 9876543210 to book an appointment.",
    "doctor_qualification": "Dr. Mehta is a General Medicine doctor (MBBS) & Dr. Sharma is a General Medicine doctor and Orthopedist (MD)."
}

training_data = [
    {"intent": "doctor_availability", "patterns": ["When is Dr. Mehta available?", "Doctor timings", "Availability of doctors", "Can I see Dr. Sharma?", "Doctor schedule"]},
    {"intent": "doctor_qualification", "patterns": ["Doctor qualification", "Doctor education", "Doctor specialisation", "Which doctor is a specialist?", "Doctors degrees"]},
    {"intent": "appointment", "patterns": ["I want to book an appointment", "Schedule appointment", "How to book a slot?", "Can I visit?", "Book a consultation"]},
    {"intent": "fees", "patterns": ["What is the consultation fee?", "How much does it cost?", "Price for visit","what is the fee", "Fee details"]},
    {"intent": "contact", "patterns": ["Contact number", "Phone number", "Call clinic", "How to reach you?"]},
    {"intent": "location", "patterns": ["Where is the clinic?", "Clinic address", "Location details", "How to reach clinic?"]},
    {"intent": "timing", "patterns": ["When will clinic open?", "timings of clinic","Open timings", "Closing time", "Working hours"]},
    {"intent": "sunday", "patterns": ["Are you open on Sunday?", "Sunday timings", "Open on Sunday?"]}
]

# ---------------------
# NLP preprocessing
# ---------------------
patterns = []
labels = []

for data in training_data:
    for pattern in data["patterns"]:
        patterns.append(pattern)
        labels.append(data["intent"])

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

patterns = [preprocess(p) for p in patterns]

# ---------------------
# Vectorization and model training
# ---------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels

clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Clinic Chatbot", page_icon="🤖")
st.title("🤖 Smart Clinic Chatbot")
st.write("Ask me anything about the clinic: timings, doctor availability, fees, appointments, and more!")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Ensure the text input key exists in session_state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Callback function for the Send button
def send_message():
    # This callback runs in Streamlit's safe callback context,
    # so modifying st.session_state here is allowed.
    msg = st.session_state.get('user_input', '').strip()
    if not msg:
        return

    processed_input = preprocess(msg)
    vect_input = vectorizer.transform([processed_input])
    intent = clf.predict(vect_input)[0]
    bot_response = clinic_data.get(intent, "Sorry, I didn’t understand that. Please try asking differently.")

    # Update history and clear the input
    st.session_state.history.append(("You", msg))
    st.session_state.history.append(("Bot", bot_response))

    # Clear the input field by setting the session_state value (safe inside callback)
    st.session_state['user_input'] = ""

# Render the input box (controlled by session_state) and a Send button that calls the callback
st.text_input("You:", key="user_input", placeholder="Type your question here...")
st.button("Send", on_click=send_message)

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")


