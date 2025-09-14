import streamlit as st
from transformers import pipeline
from langdetect import detect, DetectorFactory
import re
import html

# make langdetect deterministic
DetectorFactory.seed = 0

st.set_page_config(page_title="🇮🇳 College Info Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Multilingual College Chatbot (MarianMT + KB)")

# ---------------------------
# Knowledge base (user-provided)
# ---------------------------
RESPONSES = {
  "fees": {
    "en": "Semester fees is ₹15000. You can pay online.",
    "ta": "செமஸ்டர் கட்டணம் ₹15000. ஆன்லைனில் செலுத்தலாம்.",
    "hi": "सेमेस्टर की फीस ₹15000 है। आप ऑनलाइन भुगतान कर सकते हैं।",
    "tanglish": "Semester fees ₹15000 da 😎. Online la pay panna mudiyum."
  },
  "timetable": {
    "en": "Here is the timetable 👉 http://college.com/timetable",
    "ta": "இங்கே டைம் டேபிள் 👉 http://college.com/timetable",
    "hi": "यहाँ टाइमटेबल है 👉 http://college.com/timetable",
    "tanglish": "Inga da timetable link 👉 http://college.com/timetable"
  },
  "admission": {
    "en": "Admission process will start next month. Apply here 👉 http://college.com/admission",
    "ta": "சேர்க்கை செயல்முறை அடுத்த மாதம் தொடங்கும். இங்கே விண்ணப்பியுங்கள் 👉 http://college.com/admission",
    "hi": "प्रवेश प्रक्रिया अगले महीने शुरू होगी। यहाँ आवेदन करें 👉 http://college.com/admission",
    "tanglish": "Admission process next month start aagum da. Apply pannunga 👉 http://college.com/admission"
  },
  "hostel": {
    "en": "Yes, hostel facilities are available for both boys and girls.",
    "ta": "ஆம், ஆண்கள் மற்றும் பெண்களுக்கு விடுதி வசதிகள் உள்ளன.",
    "hi": "हाँ, लड़कों और लड़कियों दोनों के लिए हॉस्टल की सुविधा उपलब्ध है।",
    "tanglish": "Hostel iruku da boys and girls ku rendu perukum 😇."
  },
  "library": {
    "en": "Library is open from 9 AM to 6 PM on weekdays.",
    "ta": "நூலகம் வார நாட்களில் காலை 9 மணி முதல் மாலை 6 மணி வரை திறந்திருக்கும்.",
    "hi": "लाइब्रेरी सप्ताह के दिनों में सुबह 9 बजे से शाम 6 बजे तक खुली रहती है।",
    "tanglish": "Library 9 AM la open agum, 6 PM varaikum irukum weekdays la 📚."
  },
  "bus": {
    "en": "College bus service is available for all major city routes.",
    "ta": "முக்கிய நகர வழித்தடங்களுக்கு கல்லூரி பேருந்து சேவை உள்ளது.",
    "hi": "कॉलेज बस सेवा सभी प्रमुख शहर मार्गों के लिए उपलब्ध है।",
    "tanglish": "College bus major city routes ku available da 🚌."
  },
  "canteen": {
    "en": "Canteen provides fresh and affordable food for students.",
    "ta": "மாணவர்களுக்கு புது மற்றும் மலிவு உணவு உணவகம் வழங்குகிறது.",
    "hi": "कैंटीन छात्रों को ताज़ा और किफायती खाना उपलब्ध कराती है।",
    "tanglish": "Canteen la fresh ahum cheap ahum food kadaikidum da 🍔."
  },
  "results": {
    "en": "Results will be published online on the college website.",
    "ta": "முடிவுகள் கல்லூரி இணையதளத்தில் ஆன்லைனில் வெளியிடப்படும்.",
    "hi": "परिणाम कॉलेज वेबसाइट पर ऑनलाइन प्रकाशित किए जाएंगे।",
    "tanglish": "Results online la publish panniduvanga da college site la 📢."
  },
  "exam": {
    "en": "Semester exams will begin from December 10th.",
    "ta": "செமஸ்டர் தேர்வுகள் டிசம்பர் 10 முதல் தொடங்கும்.",
    "hi": "सेमेस्टर की परीक्षाएँ 10 दिसंबर से शुरू होंगी।",
    "tanglish": "Semester exam December 10th la start agum da ✍️."
  }
}

# ---------------------------
# Supported Marian models map for translation to/from English
# (only load when needed and cached)
# ---------------------------
MARIAN_TO_EN = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "ta": "Helsinki-NLP/opus-mt-ta-en",
    "te": "Helsinki-NLP/opus-mt-te-en",
    "kn": "Helsinki-NLP/opus-mt-kn-en",
    "ml": "Helsinki-NLP/opus-mt-ml-en",
    "bn": "Helsinki-NLP/opus-mt-bn-en",
    "gu": "Helsinki-NLP/opus-mt-gu-en",
    "mr": "Helsinki-NLP/opus-mt-mr-en",
    "pa": "Helsinki-NLP/opus-mt-pa-en",
    "ur": "Helsinki-NLP/opus-mt-ur-en",
    "or": "Helsinki-NLP/opus-mt-or-en",
    "as": "Helsinki-NLP/opus-mt-as-en"
}

MARIAN_FROM_EN = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
    "ml": "Helsinki-NLP/opus-mt-en-ml",
    "bn": "Helsinki-NLP/opus-mt-en-bn",
    "gu": "Helsinki-NLP/opus-mt-en-gu",
    "mr": "Helsinki-NLP/opus-mt-en-mr",
    "pa": "Helsinki-NLP/opus-mt-en-pa",
    "ur": "Helsinki-NLP/opus-mt-en-ur",
    "or": "Helsinki-NLP/opus-mt-en-or",
    "as": "Helsinki-NLP/opus-mt-en-as"
}

# cache translation pipelines
@st.cache_resource
def get_translation_pipeline(model_name):
    return pipeline("translation", model=model_name)

# ---------------------------
# Intent keywords (English) for simple mapping
# ---------------------------
INTENT_KEYWORDS = {
    "fees": ["fee", "fees", "semester fee", "tuition", "pay", "price", "kattam", "fee?"],
    "timetable": ["timetable", "time table", "schedule", "class timing", "timetable"],
    "admission": ["admission", "apply", "application", "enroll", "entry"],
    "hostel": ["hostel", "dorm", "accommodation", "stay"],
    "library": ["library", "books", "library timing", "reading room"],
    "bus": ["bus", "transport", "shuttle", "route"],
    "canteen": ["canteen", "food", "mess", "snack"],
    "results": ["result", "results", "marks", "grades"],
    "exam": ["exam", "exams", "semester exam", "test", "hall ticket"]
}

# flatten keywords to allow quick check
ALL_KEYWORDS = []
for klist in INTENT_KEYWORDS.values():
    ALL_KEYWORDS += klist

# ---------------------------
# Helper functions
# ---------------------------
def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = s.lower().strip()
    s = re.sub(r"http\S+", " ", s)  # remove urls for matching
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text: str, lang: str) -> str:
    """If we have a Marian model for lang->en, use it, else return original text."""
    if lang == "en":
        return text
    model_name = MARIAN_TO_EN.get(lang)
    if not model_name:
        return text
    pipe = get_translation_pipeline(model_name)
    out = pipe(text, max_length=512)
    # some pipelines return list of dicts
    if isinstance(out, list):
        return out[0].get("translation_text", str(out[0]))
    if isinstance(out, dict):
        return out.get("translation_text", str(out))
    return str(out)

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English text to target_lang if model exists, else return English."""
    if target_lang == "en":
        return text
    model_name = MARIAN_FROM_EN.get(target_lang)
    if not model_name:
        return text
    pipe = get_translation_pipeline(model_name)
    out = pipe(text, max_length=512)
    if isinstance(out, list):
        return out[0].get("translation_text", str(out[0]))
    if isinstance(out, dict):
        return out.get("translation_text", str(out))
    return str(out)

def match_intent(english_text: str) -> str:
    """Simple keyword-based intent classification returning one of RESPONSES keys or None."""
    txt = clean_text(english_text)
    # direct keyword check (longest match first)
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                return intent
    # fallback: check single-word contains any keyword substring
    words = txt.split()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            for w in words:
                if kw in w or w in kw:
                    return intent
    return None

# ---------------------------
# UI
# ---------------------------
st.markdown("**Instructions:** Type your question in any Indian language (e.g., Tamil, Hindi, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, Punjabi, Urdu, Odia, Assamese, or English).")
use_tanglish = st.checkbox("Prefer Tanglish responses (if available)", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user input
if user_input := st.chat_input("Ask about fees, timetable, admission, hostel, library, bus, canteen, results, exam..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # detect language
    lang = detect_language_safe(user_input)
    # if detect returns something like 'en' or 'ta' etc.
    # translate to English for intent detection (if model available)
    english_text = translate_to_english(user_input, lang)

    # intent detection
    intent = match_intent(english_text)

    if intent:
        # choose the best language key for reply
        reply_text = None
        # if user prefers tanglish and tanglish exists
        if use_tanglish and "tanglish" in RESPONSES[intent]:
            reply_text = RESPONSES[intent]["tanglish"]
        # if exact language exists in KB
        elif lang in RESPONSES[intent]:
            reply_text = RESPONSES[intent][lang]
        # fallback to english in KB
        elif "en" in RESPONSES[intent]:
            # if requested language not in KB, translate the English KB reply into user language (if possible)
            en_reply = RESPONSES[intent]["en"]
            translated = translate_from_english(en_reply, lang)
            reply_text = translated if translated else en_reply
        else:
            reply_text = "Sorry, I don't have that information right now."

    else:
        # no intent matched: give polite fallback
        # We attempt a fuzzy fallback by scanning for small hints (e.g., numbers, 'admission', urls)
        fallback = "Sorry da, mudiyala exact-a puriyala. Please ask about fees, timetable, admission, hostel, library, bus, canteen, results, or exam."
        # try to return fallback in user's language if model exists
        reply_text = translate_from_english(fallback, lang)

    # append & display
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
