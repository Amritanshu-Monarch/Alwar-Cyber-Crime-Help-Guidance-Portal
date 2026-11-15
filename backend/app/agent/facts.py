"""Authoritative Single Source of Truth for verified static facts, helplines, and local police data.
All components (tools, prompts, RAG, and UI) read from or align with this module.
"""

CONTROL_ROOM = "8764874306"

# Authoritative facts registry with metadata
OFFICIAL_FACTS = {
    "national_cyber_helpline": {
        "value": "1930",
        "label": "National Cyber Crime Helpline (Golden Hour)",
        "source": "Ministry of Home Affairs / I4C",
        "authority": "official_government",
        "jurisdiction": "national",
        "last_verified": "2026-11-15",
    },
    "alwar_cyber_control_room": {
        "value": CONTROL_ROOM,
        "label": "Alwar Cyber Control Room",
        "source": "Rajasthan Police Alwar District",
        "authority": "official_rajasthan_police",
        "jurisdiction": "alwar_local",
        "last_verified": "2026-11-15",
    },
    "police_emergency": {
        "value": "112",
        "label": "Police Emergency (24x7)",
        "source": "Government of India / ERSS",
        "authority": "official_government",
        "jurisdiction": "national",
        "last_verified": "2026-11-15",
    },
    "women_helpline": {
        "value": "1090",
        "label": "Women Helpline (Harassment / Safety)",
        "source": "Rajasthan Police",
        "authority": "official_rajasthan_police",
        "jurisdiction": "rajasthan",
        "last_verified": "2026-11-15",
    },
    "alwar_sp_office_phone": {
        "value": "0144-2337456",
        "label": "Superintendent of Police Office, Alwar",
        "source": "Rajasthan Police Alwar",
        "authority": "official_rajasthan_police",
        "jurisdiction": "alwar_local",
        "last_verified": "2026-11-15",
    },
    "alwar_ccps_email": {
        "value": "ccps.alwar@rajpolice.gov.in",
        "label": "Cyber Crime Police Station (CCPS) Email",
        "source": "Rajasthan Police Alwar",
        "authority": "official_rajasthan_police",
        "jurisdiction": "alwar_local",
        "last_verified": "2026-11-15",
    },
}

CONTACTS = {
    "alwar_cyber_control_room": {"label": OFFICIAL_FACTS["alwar_cyber_control_room"]["label"], "phone": CONTROL_ROOM},
    "national_cyber_helpline": {"label": OFFICIAL_FACTS["national_cyber_helpline"]["label"], "phone": "1930"},
    "police_emergency": {"label": OFFICIAL_FACTS["police_emergency"]["label"], "phone": "112"},
    "women_helpline": {"label": OFFICIAL_FACTS["women_helpline"]["label"], "phone": "1090"},
    "links": {
        "national_cyber_portal": "https://cybercrime.gov.in",
        "chakshu_report_fake_call_sms": "https://sancharsaathi.gov.in/sfc/Home/sfc-complaint.jsp",
        "ceir_block_stolen_mobile": "https://sancharsaathi.gov.in",
        "whatsapp_cyber_desk": f"https://wa.me/91{CONTROL_ROOM}",
        "rajasthan_police_citizen_services": "https://police.rajasthan.gov.in",
    },
}

# Verified police stations in Alwar district and police commissionerate/ranges
STATIONS = [
    {"name": "Kotwali Police Station", "area": "Hope Circus, Purana Katla, Munshi Bagh, Church Road, City Center"},
    {"name": "NEB Police Station", "area": "Naya Bass, Hasan Khan Mewati Nagar, Ambedkar Nagar, Transport Nagar"},
    {"name": "Aravali Vihar Police Station", "area": "Kala Kuan, Housing Board, Surya Nagar, Scheme 1, 2, 8"},
    {"name": "Shivaji Park Police Station", "area": "Shivaji Park, Malviya Nagar, Scheme 10, Shanti Kunj"},
    {"name": "Mahila Thana Alwar", "area": "Entire Alwar District (Crimes against women & harassment)"},
    {"name": "Sadar Police Station, Alwar", "area": "Alwar Rural, Daudpur, Delhi Road bypass, Jaipur Road"},
    {"name": "Udyog Nagar Police Station", "area": "Matsya Industrial Area (MIA), RIICO, Delhi-Jaipur highway"},
    {"name": "Cyber Crime Police Station (CCPS), Alwar", "area": "SP Office Campus, Alwar (District Cyber Cell & technical forensics)"},
    {"name": "Bhiwadi (Phoolbagh) Police Station", "area": "Bhiwadi industrial area, Phoolbagh"},
    {"name": "Tijara Police Station", "area": "Tijara town and rural sub-division"},
    {"name": "Kishangarh Bas Police Station", "area": "Kishangarh Bas, Khairthal Road"},
    {"name": "Behror Police Station", "area": "Behror town, NH-48 corridor"},
    {"name": "Neemrana Police Station", "area": "Neemrana, Japanese Zone, RIICO Phase 1-3"},
    {"name": "Shahjahanpur Police Station", "area": "Shahjahanpur, Rajasthan-Haryana border"},
    {"name": "Ramgarh Police Station", "area": "Ramgarh tehsil and rural circle"},
    {"name": "Laxmangarh Police Station", "area": "Laxmangarh sub-division"},
    {"name": "Rajgarh Police Station", "area": "Rajgarh town, Railway station area"},
    {"name": "Thanagazi Police Station", "area": "Thanagazi, Sariska border corridor"},
]

# Mapping common localities / landmarks to their jurisdictional Thana
LOCALITY_TO_STATION = {
    "kala kuan": "Aravali Vihar Police Station",
    "housing board": "Aravali Vihar Police Station",
    "surya nagar": "Aravali Vihar Police Station",
    "scheme 1": "Aravali Vihar Police Station",
    "scheme 2": "Aravali Vihar Police Station",
    "scheme 8": "Aravali Vihar Police Station",
    "hope circus": "Kotwali Police Station",
    "purana katla": "Kotwali Police Station",
    "munshi bagh": "Kotwali Police Station",
    "church road": "Kotwali Police Station",
    "city center": "Kotwali Police Station",
    "naya bass": "NEB Police Station",
    "neb": "NEB Police Station",
    "hasan khan": "NEB Police Station",
    "hasan khan mewati": "NEB Police Station",
    "transport nagar": "NEB Police Station",
    "ambedkar nagar": "NEB Police Station",
    "shivaji park": "Shivaji Park Police Station",
    "malviya nagar": "Shivaji Park Police Station",
    "scheme 10": "Shivaji Park Police Station",
    "mia": "Udyog Nagar Police Station",
    "matsya industrial": "Udyog Nagar Police Station",
    "daudpur": "Sadar Police Station, Alwar",
    "delhi road": "Sadar Police Station, Alwar",
    "bhiwadi": "Bhiwadi (Phoolbagh) Police Station",
    "tijara": "Tijara Police Station",
    "kishangarh bas": "Kishangarh Bas Police Station",
    "behror": "Behror Police Station",
    "neemrana": "Neemrana Police Station",
    "shahjahanpur": "Shahjahanpur Police Station",
    "ramgarh": "Ramgarh Police Station",
    "laxmangarh": "Laxmangarh Police Station",
    "rajgarh": "Rajgarh Police Station",
    "thanagazi": "Thanagazi Police Station",
}

MAPS_URL = "https://maps.google.com/?q="

INCIDENT_CATEGORIES = [
    "Online Financial Fraud / OTP Theft",
    "WhatsApp / Fake Call Impersonation",
    "Social Media Harassment / Fake Profile",
    "Deepfake / Morphed Media Harassment",
    "Sextortion / Video Call Blackmail",
    "Deepfake & Financial Extortion",
    "ATM / Credit Card Fraud",
    "Job / Loan Scam",
    "Hacked Account / Unauthorized Access",
    "Other Cyber Crime",
]

# Hindi display translations for police stations
STATION_NAME_HI = {
    "Kotwali Police Station": "कोतवाली पुलिस थाना",
    "NEB Police Station": "एनईबी पुलिस थाना",
    "Aravali Vihar Police Station": "अरावली विहार पुलिस थाना",
    "Shivaji Park Police Station": "शिवाजी पार्क पुलिस थाना",
    "Mahila Thana Alwar": "महिला थाना अलवर",
    "Sadar Police Station, Alwar": "सदर पुलिस थाना, अलवर",
    "Udyog Nagar Police Station": "उद्योग नगर पुलिस थाना",
    "Cyber Crime Police Station (CCPS), Alwar": "साइबर क्राइम पुलिस थाना (CCPS), अलवर",
    "Bhiwadi (Phoolbagh) Police Station": "भिवाड़ी (फूलबाग) पुलिस थाना",
    "Tijara Police Station": "तिजारा पुलिस थाना",
    "Kishangarh Bas Police Station": "किशनगढ़ बास पुलिस थाना",
    "Behror Police Station": "बहरोड़ पुलिस थाना",
    "Neemrana Police Station": "नीमराना पुलिस थाना",
    "Shahjahanpur Police Station": "शाहजहांपुर पुलिस थाना",
    "Ramgarh Police Station": "रामगढ़ पुलिस थाना",
    "Laxmangarh Police Station": "लक्ष्मणगढ़ पुलिस थाना",
    "Rajgarh Police Station": "राजगढ़ पुलिस थाना",
    "Thanagazi Police Station": "थानागाजी पुलिस थाना",
}

# Hindi display translations for complaint categories
CATEGORY_HI = {
    "Online Financial Fraud / OTP Theft": "ऑनलाइन वित्तीय धोखाधड़ी / अनधिकृत निकासी",
    "WhatsApp / Fake Call Impersonation": "व्हाट्सएप / फर्जी कॉल पहचान चोरी",
    "Social Media Harassment / Fake Profile": "सोशल मीडिया उत्पीड़न / फर्जी प्रोफाइल",
    "Deepfake / Morphed Media Harassment": "डीपफेक / मॉर्फ्ड फोटो-वीडियो उत्पीड़न",
    "Sextortion / Video Call Blackmail": "सेक्सटॉर्शन / वीडियो कॉल ब्लैकमेल",
    "Deepfake & Financial Extortion": "डीपफेक एवं वित्तीय जबरन वसूली",
    "ATM / Credit Card Fraud": "एटीएम / क्रेडिट कार्ड धोखाधड़ी",
    "Job / Loan Scam": "नौकरी / लोन धोखाधड़ी",
    "Hacked Account / Unauthorized Access": "व्हाट्सएप / सोशल मीडिया अकाउंट हैकिंग एवं अनधिकृत एक्सेस",
    "Other Cyber Crime": "साइबर अपराध",
    "Other": "साइबर अपराध",
}

