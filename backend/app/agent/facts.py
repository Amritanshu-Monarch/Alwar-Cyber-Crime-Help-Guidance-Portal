"""Verified static facts from the portal. Edit here; tools and prompt read from this."""

CONTROL_ROOM = "8764874306"

CONTACTS = {
    "alwar_cyber_control_room": {"label": "Alwar Cyber Control Room", "phone": CONTROL_ROOM},
    "national_cyber_helpline": {"label": "National Cyber Crime Helpline (Golden Hour)", "phone": "1930"},
    "police_emergency": {"label": "Police Emergency (24x7)", "phone": "112"},
    "links": {
        "national_cyber_portal": "https://cybercrime.gov.in",
        "chakshu_report_fake_call_sms": "https://sancharsaathi.gov.in/sfc/Home/sfc-complaint.jsp",
        "whatsapp_cyber_desk": f"https://wa.me/91{CONTROL_ROOM}",
        "rajasthan_police_citizen_services": "https://police.rajasthan.gov.in",
    },
}

STATIONS = [
    {"name": "Kotwali Police Station", "area": "Alwar City"},
    {"name": "NEB Police Station", "area": "Alwar City"},
    {"name": "Aravali Vihar Thana", "area": "Alwar"},
    {"name": "Mahila Thana Alwar", "area": "Alwar (women's desk)"},
]

MAPS_URL = "https://maps.google.com/?q="

INCIDENT_CATEGORIES = [
    "Online Financial Fraud / OTP Theft",
    "WhatsApp / Fake Call Impersonation",
    "Social Media Harassment / Fake Profile",
    "ATM / Credit Card Fraud",
    "Job / Loan Scam",
    "Other",
]
