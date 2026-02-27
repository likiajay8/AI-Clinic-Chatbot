🏥 AI Clinic Chatbot

An intelligent clinic support chatbot built with Streamlit and NLP that helps users quickly get information about clinic services such as timings, doctor availability, fees, and appointments.

🔗 Live Demo: https://ai-clinic-chatbot-d4ywcdburgxhbkyrgssyym.streamlit.app/

🚀 Features

💬 Understands common clinic queries using NLP

🧠 Intent classification using Logistic Regression

🕒 Provides clinic timings and doctor availability

📍 Shares location and contact details

🌐 Interactive web interface built with Streamlit

🔁 Chat history support

🛠 Tech Stack

Python

Streamlit

NLTK (Natural Language Processing)

Scikit-learn

Machine Learning (Logistic Regression)

⚙️ How It Works

User enters a query in the chat interface

Text is preprocessed (lowercasing, punctuation removal, lemmatization)

Query is converted to vectors using CountVectorizer

Logistic Regression model predicts the intent

Bot returns the most relevant clinic response

🧠 Supported Intents

Clinic timings

Sunday availability

Doctor availability

Doctor qualifications

Appointment booking

Consultation fees

Contact details

Clinic location

📂 Project Structure

ai-clinic-chatbot/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

▶️ Run Locally

# Clone the repository
git clone https://github.com/your-username/ai-clinic-chatbot.git

# Navigate to project folder
cd ai-clinic-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📸 Demo

You can try the chatbot live here:
👉 https://ai-clinic-chatbot-d4ywcdburgxhbkyrgssyym.streamlit.app/

🎯 Use Cases

Clinic websites for quick patient assistance

Healthcare information kiosks

Customer support automation

NLP learning project

🔮 Future Improvements

Add more medical FAQs

Integrate real database for appointments

Add speech-to-text support

Deploy with backend API

Improve model with deep learning

👨‍💻 Author

Likith H P
