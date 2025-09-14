import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langdetect import detect, DetectorFactory
import re, html, time

# deterministic langdetect
DetectorFactory.seed = 0

st.set_page_config(page_title="🇮🇳 College Info Chatbot", page_icon="🤖", layout="centered")
st.title("Conversa AI-Multilingual")

# ---------------------------
# Knowledge base (user-provided)
# ---------------------------
RESPONSES = {
  "fees": {
    "en": "Semester fees is ₹15000. You can pay online.",
    "ta": "Semester fees ₹15000. Online la pay panna mudiyum.",
    "hi": "semestar kee phees ₹15000 hai. aap onalain bhugataan kar sakate hain.",
    "tanglish": "Semester fees ₹15000 da 😎. Online la pay panna mudiyum.",
    "kn": "Semester fees ₹15000. Nivu online mEle pay maadabahudu",
    "ml": "Semester fees ₹15000. Nee online-il pay cheyyam",
    "bn": "Semester fees ₹15000. Apni online pay korte paren",
    "gu": "Semester fees ₹15000. Tame online pay kari shako cho",
    "mr": "Semester fees ₹15000. Tumhi online pay karu shakta",
    "pa": "Semester fees ₹15000. Tusi online pay kar sakde ho",
    "ur": "Semester fees ₹15000. Aap online pay kar saktay hain",
    "or": "Semester fees ₹15000. Apan online pay kariparibe",
    "as": "Semester fees ₹15000. Apuni online pay koribo paribo"
  },
  "timetable": {
    "en": "Here is the timetable 👉 http://college.com/timetable",
    "ta": "இங்கே டைம் டேபிள் 👉 http://college.com/timetable",
    "hi": "yahaan taimatebal hai 👉 http://college.com/timetable",
    "tanglish": "Inga da timetable link 👉 http://college.com/timetable",
    "kn": "Ivatthu timetable idhe 👉 http://college.com/timetable",
    "ml": "Ivide timetable aanu 👉 http://college.com/timetable",
    "bn": "Ekhane timetable ache 👉 http://college.com/timetable",
    "gu": "Ahiyan timetable chhe 👉 http://college.com/timetable",
    "mr": "Ithe timetable aahe 👉 http://college.com/timetable",
    "pa": "Ithe timetable hai 👉 http://college.com/timetable",
    "ur": "Yahaan timetable hai 👉 http://college.com/timetable",
    "or": "Ethe timetable achhi 👉 http://college.com/timetable",
    "as": "Ekhane timetable ase 👉 http://college.com/timetable"
  },
  "admission": {
    "en": "Admission process will start next month. Apply here 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "ta": "சேர்க்கை செயல்முறை அடுத்த மாதம் தொடங்கும். இங்கே விண்ணப்பியுங்கள் 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "hi": "pravesh prakriya agale maheene shuroo hogee. yahaan aavedan karen 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "tanglish": "Admission process next month start aagum da. Apply pannunga 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "kn": "Next month admission process aarambhavagutte. Ille apply maadi 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "ml": "Next month admission process thudangum. Ivide apply cheyyuka 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "bn": "Agami maashe admission process shuru hobe. Ekhane apply korun 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "gu": "Aavti maheena maadhe admission process sharu thashe. Ahiyan apply karo 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "mr": "Pudhil mahinyat admission process suru hoil. Ithe apply kara 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "pa": "Agla mahina admission process shuru hovega. Ithe apply karo 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "ur": "Agla mahina admission process shuru hogi. Yahaan apply karein 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "or": "Aagami maas re admission process suru heba. Ethe apply karantu 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/",
    "as": "Agami maasot admission process suru hobo. Ekhane apply korun 👉 https://www.srcas.ac.in/admission/admission-policy-and-process/"
  },
  "hostel": {
    "en": "Yes, hostel facilities are available for both boys and girls.",
    "ta": "ஆம், ஆண்கள் மற்றும் பெண்களுக்கு விடுதி வசதிகள் உள்ளன.",
    "hi": "pravesh prakriya agale maheene shuroo hogee. yahaan aavedan karen",
    "tanglish": "Hostel iruku da boys and girls ku rendu perukum 😇.",
    "kn": "Ho hostel suvidhagala both boys matthu girls ge available ide.",
    "ml": "Athu hostel suvidhakal randu boysum girlsum available aanu.",
    "bn": "Haan hostel facilities dono boys ebong girls er jonno available ache.",
    "gu": "Ha hostel facilities boys ane girls mate available chhe.",
    "mr": "Ho hostel suvidha boys ani girls sathi available ahe.",
    "pa": "Haan hostel facilities dono boys te girls layi available han.",
    "ur": "Haan hostel facilities dono boys aur girls ke liye available hain.",
    "or": "Ho hostel suvidhaguloo boys o girls pain available achhi.",
    "as": "Ho hostel suvidha dono boys aru girls karone available ase."
  },
  "library": {
    "en": "Library is open from 9 AM to 6 PM on weekdays.",
    "ta": "நூலகம் வார நாட்களில் காலை 9 மணி முதல் மாலை 6 மணி வரை திறந்திருக்கும்.",
    "hi": "laibreree saptaah ke dinon mein subah 9 baje se shaam 6 baje tak khulee rahatee hai.",
    "tanglish": "Library 9 AM la open agum, 6 PM varaikum irukum weekdays la 📚.",
    "kn": "Library 9 AM inda 6 PM vare weekdays ge open ide.",
    "ml": "Library 9 AM ninnu 6 PM vare weekdays-il open aanu.",
    "bn": "Library 9 AM theke 6 PM porjonto weekdays e open ache.",
    "gu": "Library 9 AM thi 6 PM sudhi weekdays ma open chhe.",
    "mr": "Library 9 AM pasun 6 PM paryant weekdays sathi open ahe.",
    "pa": "Library 9 AM ton 6 PM tak weekdays layi open han.",
    "ur": "Library 9 AM se 6 PM tak weekdays ke liye open hai.",
    "or": "Library 9 AM ru 6 PM paryanta weekdays re open achhi.",
    "as": "Library 9 AM pora 6 PM porjonto weekdays khula ase."
  },
  "bus": {
    "en": "College bus service is available for all major city routes.",
    "ta": "முக்கிய நகர வழித்தடங்களுக்கு கல்லூரி பேருந்து சேவை உள்ளது.",
    "hi": "kolej bas seva sabhee pramukh shahar maargon ke lie upalabdh hai.",
    "tanglish": "College bus major city routes ku available da 🚌.",
    "kn": "College bus service ella major city routes ge available ide.",
    "ml": "College bus service ella major city routes-il available aanu.",
    "bn": "College bus service shob major city routes er jonno available ache.",
    "gu": "College bus service badha major city routes mate available chhe.",
    "mr": "College bus service saglya major city routes sathi available ahe.",
    "pa": "College bus service sare major city routes layi available han.",
    "ur": "College bus service sab major city routes ke liye available hain.",
    "or": "College bus service samasta major city routes pain available achhi.",
    "as": "College bus service sob major city routes karone available ase."
  },
  "canteen": {
    "en": "Canteen provides fresh and affordable food for students.",
    "ta": "மாணவர்களுக்கு புது மற்றும் மலிவு உணவு உணவகம் வழங்குகிறது.",
    "hi": "kainteen chhaatron ko taaza aur kiphaayatee khaana upalabdh karaatee hai.",
    "tanglish": "Canteen la fresh ahum cheap ahum food kadaikidum da 🍔.",
    "kn": "Canteen students ge fresh mattu affordable food provide maadutte.",
    "ml": "Canteen studentsinu fresh um affordable um food provide cheyyunnu.",
    "bn": "Canteen students er jonno fresh ebong affordable food provide kore.",
    "gu": "Canteen students mate fresh ane affordable food provide kare chhe.",
    "mr": "Canteen students sathi fresh ani affordable food provide karte.",
    "pa": "Canteen students layi fresh te affordable food provide karda hai.",
    "ur": "Canteen students ke liye fresh aur affordable food provide karta hai.",
    "or": "Canteen students pain fresh o affordable food provide karuchi.",
    "as": "Canteen students karone fresh aru affordable food provide kore."
  },
  "results": {
    "en": "Results will be published online on the college website.",
    "ta": "முடிவுகள் கல்லூரி இணையதளத்தில் ஆன்லைனில் வெளியிடப்படும்.",
    "hi": "parinaam kolej vebasait par onalain prakaashit kie jaenge.",
    "tanglish": "Results online la publish panniduvanga da college site la 📢.",
    "kn": "Results online college website mele publish aagutte.",
    "ml": "Results online college website-il publish cheyyapetum.",
    "bn": "Results online college website-e publish hobe.",
    "gu": "Results online college website par publish thashe.",
    "mr": "Results online college website var publish honaar.",
    "pa": "Results online college website te publish honge.",
    "ur": "Results online college website par publish honge.",
    "or": "Results online college website re publish heba.",
    "as": "Results online college website ot publish hobo."
  },
  "exam": {
    "en": "Semester exams will begin from December 10th.",
    "ta": "செமஸ்டர் தேர்வுகள் டிசம்பர் 10 முதல் தொடங்கும்.",
    "hi": "semestar kee pareekshaen 10 disambar se shuroo hongee.",
    "tanglish": "Semester exam December 10th la start agum da ✍️.",
    "kn": "Semester exams December 10th inda aarambhavagutte.",
    "ml": "Semester exams December 10th ninnu thudangum.",  
    "bn": "Semester exams December 10th theke shuru hobe.",
    "gu": "Semester exams December 10th thi sharu thashe.",
    "mr": "Semester exams December 10th pasun suru honaar.",
    "pa": "Semester exams December 10th ton shuru honge.",
    "ur": "Semester exams December 10th se shuru honge.",
    "or": "Semester exams December 10th ru suru heba.",
    "as": "Semester exams December 10th pora suru hobo."
  },
}
# ---------------------------
# Supported Marian models map for translation to/from English
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

# ---------------------------
# Cache translation pipelines
# ---------------------------
@st.cache_resource
def get_translation_pipeline(model_name):
    # small wrapper so pipeline is cached by model_name
    return pipeline("translation", model=model_name)

# ---------------------------
# Simple intent keywords (English) for mapping to KB
# ---------------------------
INTENT_KEYWORDS = {
    "fees": ["fee", "fees", "semester fee", "tuition", "pay", "kattam", "varga", "charges"],
    "timetable": ["timetable", "time table", "schedule", "class timing", "timetable"],
    "admission": ["admission", "apply", "application", "enroll", "entry"],
    "hostel": ["hostel", "dorm", "accommodation", "stay"],
    "library": ["library", "books", "reading", "study room"],
    "bus": ["bus", "transport", "shuttle", "route"],
    "canteen": ["canteen", "food", "mess", "snack"],
    "results": ["result", "results", "marks", "grades"],
    "exam": ["exam", "exams", "semester exam", "test", "hall ticket"]
}

def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = s.lower().strip()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language_safe(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def translate_to_english(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    model_name = MARIAN_TO_EN.get(lang)
    if not model_name:
        return text
    try:
        pipe = get_translation_pipeline(model_name)
        out = pipe(text, max_length=512)
        if isinstance(out, list):
            return out[0].get("translation_text", str(out[0]))
        if isinstance(out, dict):
            return out.get("translation_text", str(out))
        return str(out)
    except Exception as e:
        # on failure, return original text
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    if not text or target_lang == "en":
        return text
    model_name = MARIAN_FROM_EN.get(target_lang)
    if not model_name:
        return text
    try:
        pipe = get_translation_pipeline(model_name)
        out = pipe(text, max_length=512)
        if isinstance(out, list):
            return out[0].get("translation_text", str(out[0]))
        if isinstance(out, dict):
            return out.get("translation_text", str(out))
        return str(out)
    except Exception as e:
        return text

def match_intent(english_text: str) -> str:
    txt = clean_text(english_text)
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                return intent
    # fallback: word-level fuzzy
    words = txt.split()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            for w in words:
                if kw in w or w in kw:
                    return intent
    return None

# ---------------------------
# Simple English fallback chatbot model (DialoGPT-small)
# ---------------------------
@st.cache_resource
def load_fallback_chatbot():
    model_name = "microsoft/DialoGPT-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tok, model

# load once (may download ~200-400MB)
with st.spinner("Loading fallback chatbot model (first run may take time)..."):
    try:
        tokenizer, chatbot_model = load_fallback_chatbot()
    except Exception as e:
        st.error("Failed to load fallback chatbot model: " + str(e))
        tokenizer, chatbot_model = None, None

def chatbot_response(prompt: str) -> str:
    if tokenizer is None or chatbot_model is None:
        return "Sorry, chatbot brain not available right now."
    try:
        inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        outputs = chatbot_model.generate(inputs, max_length=300, pad_token_id=tokenizer.eos_token_id)
        # decode only the generated part (after the input)
        reply = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return reply.strip()
    except Exception as e:
        return "Sorry, failed to generate reply."

# ---------------------------
# UI & Chat loop
# ---------------------------
st.markdown("**Instructions:** Type your question in any Indian language (e.g., Tamil, Hindi, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, Punjabi, Urdu, Odia, Assamese) or in English.")
use_tanglish = st.checkbox("Prefer Tanglish responses (if available)", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# input
if user_input := st.chat_input("Ask about fees, timetable, admission, hostel, library, bus, canteen, results, exam..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # detect language
    lang = detect_language_safe(user_input)  # lang codes like 'ta', 'hi', 'en' etc.
    # for some languages detect gives 'bn' etc. we already handle many mappings

    # translate to english for intent detection
    english_text = translate_to_english(user_input, lang)

    # detect intent
    intent = match_intent(english_text)

    # prepare reply
    if intent:
        # prefer tanglish if user asked and KB has tanglish
        if use_tanglish and "tanglish" in RESPONSES[intent]:
            reply_text = RESPONSES[intent]["tanglish"]
        # if KB has exact language reply (ta/hi/en) use it
        elif lang in RESPONSES[intent]:
            reply_text = RESPONSES[intent][lang]
        else:
            # fallback: translate KB english reply to user's language (if model exist)
            en_reply = RESPONSES[intent].get("en", "")
            translated = translate_from_english(en_reply, lang)
            reply_text = translated if translated else en_reply
    else:
        # no KB intent matched -> use fallback chatbot
        # get english fallback prompt
        fallback_prompt = english_text if english_text.strip() else "Hello"
        bot_en = chatbot_response(fallback_prompt)
        # translate back to user language if possible
        bot_translated = translate_from_english(bot_en, lang)
        reply_text = bot_translated if bot_translated else bot_en

    # append & display
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
