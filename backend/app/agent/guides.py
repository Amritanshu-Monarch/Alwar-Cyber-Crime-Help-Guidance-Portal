"""Deterministic step-by-step complaint / application checklists (en + hi).
Kept as data, not LLM output, so the steps people follow are exactly what we reviewed.
Facts here must match kb/*.md."""

LABELS = {
    "en": {
        "where": "Where and how to report",
        "documents": "Documents to keep ready",
        "tell": "What to tell",
        "helplines": "Helplines",
    },
    "hi": {
        "where": "कहाँ और कैसे शिकायत करें",
        "documents": "तैयार रखने वाले दस्तावेज़",
        "tell": "क्या बताएं",
        "helplines": "हेल्पलाइन",
    },
}

HELP_EN = ["1930 National Cyber Crime Helpline", "8764874306 Alwar Cyber Control Room", "112 Police Emergency"]
HELP_HI = ["1930 राष्ट्रीय साइबर क्राइम हेल्पलाइन", "8764874306 अलवर साइबर कंट्रोल रूम", "112 पुलिस आपातकाल"]

GUIDES: dict[str, dict[str, dict]] = {
    "financial_fraud": {
        "en": {
            "title": "Online financial fraud (money lost via UPI, OTP, card)",
            "where": [
                "Call 1930 immediately (Golden Hour) and note the acknowledgement number.",
                "Call your bank's customer care, block the card/UPI/net banking and raise an unauthorised transaction dispute.",
                "Register the complaint on cybercrime.gov.in and quote the 1930 acknowledgement number.",
                "Give a written complaint at the nearest police station or to the Alwar Cyber Control Room (8764874306); ask for a receipt.",
            ],
            "documents": [
                "Bank statement or SMS showing the debit; transaction ID (UTR)",
                "Screenshots of chats, call log, messages, payment page, website",
                "Fraudster's phone number, UPI ID, bank account number",
                "ID proof (Aadhaar / Voter ID); the app or APK name if you installed one (do not delete it)",
            ],
            "tell": [
                "Your name, phone, address",
                "Date, exact time and amount lost",
                "How it started: call, SMS, WhatsApp, link or app, and what the fraudster said",
                "Whether you shared an OTP/PIN or installed an app",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "ऑनलाइन वित्तीय ठगी (UPI, OTP, कार्ड से पैसे गए)",
            "where": [
                "तुरंत 1930 पर कॉल करें (गोल्डन आवर) और acknowledgement नंबर नोट करें।",
                "बैंक के कस्टमर केयर को कॉल कर कार्ड/UPI/नेट बैंकिंग ब्लॉक करवाएं और अनधिकृत लेनदेन का विवाद दर्ज कराएं।",
                "cybercrime.gov.in पर शिकायत दर्ज करें और 1930 वाला acknowledgement नंबर लिखें।",
                "नजदीकी थाने में या अलवर साइबर कंट्रोल रूम (8764874306) पर लिखित शिकायत दें और रसीद लें।",
            ],
            "documents": [
                "बैंक स्टेटमेंट या कटौती का SMS; ट्रांजैक्शन ID (UTR)",
                "चैट, कॉल लॉग, मैसेज, पेमेंट पेज और वेबसाइट के स्क्रीनशॉट",
                "ठग का फोन नंबर, UPI ID, बैंक खाता नंबर",
                "पहचान पत्र (आधार / वोटर ID); अगर कोई ऐप या APK इंस्टॉल किया तो उसका नाम (डिलीट न करें)",
            ],
            "tell": [
                "आपका नाम, फोन, पता",
                "तारीख, सटीक समय और ठगी की राशि",
                "शुरुआत कैसे हुई: कॉल, SMS, WhatsApp, लिंक या ऐप, और ठग ने क्या कहा",
                "क्या आपने OTP/PIN बताया या कोई ऐप इंस्टॉल किया",
            ],
            "helplines": HELP_HI,
        },
    },
    "account_hacking": {
        "en": {
            "title": "Hacked social media / email account",
            "where": [
                "Use the platform's account recovery, change the password, turn on two factor authentication and log out of all devices.",
                "Report the hacking to the platform's help centre.",
                "File a complaint on cybercrime.gov.in; if money or blackmail is involved also call 1930 / visit the police station.",
            ],
            "documents": [
                "Screenshots of the hacked account, login alerts, strange messages",
                "Profile link / username and the email or phone linked to it",
                "ID proof; any ransom or blackmail messages",
            ],
            "tell": [
                "Which account, when you lost access and how you noticed",
                "Any link, app or call just before it happened",
                "What the hacker did: posts, messages, money requests",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "हैक हुआ सोशल मीडिया / ईमेल अकाउंट",
            "where": [
                "प्लेटफ़ॉर्म की account recovery से अकाउंट वापस लें, पासवर्ड बदलें, two factor authentication चालू करें और सभी डिवाइस से लॉग आउट करें।",
                "प्लेटफ़ॉर्म के हेल्प सेंटर में हैकिंग रिपोर्ट करें।",
                "cybercrime.gov.in पर शिकायत करें; पैसे या ब्लैकमेल का मामला हो तो 1930 पर कॉल करें / थाने जाएं।",
            ],
            "documents": [
                "हैक हुए अकाउंट, लॉगिन अलर्ट और अजीब मैसेज के स्क्रीनशॉट",
                "प्रोफ़ाइल लिंक / यूज़रनेम और उससे जुड़ा ईमेल या फोन",
                "पहचान पत्र; फिरौती या ब्लैकमेल के मैसेज",
            ],
            "tell": [
                "कौन सा अकाउंट, कब एक्सेस गया और कैसे पता चला",
                "घटना से ठीक पहले कोई लिंक, ऐप या कॉल",
                "हैकर ने क्या किया: पोस्ट, मैसेज, पैसे की मांग",
            ],
            "helplines": HELP_HI,
        },
    },
    "harassment_or_fake_profile": {
        "en": {
            "title": "Online harassment or fake profile",
            "where": [
                "Take screenshots with profile link, username, date and time. Do not delete the chat.",
                "Report and block the profile on the platform.",
                "File a complaint on cybercrime.gov.in (women and child related crimes can be reported anonymously).",
                "Give a written complaint at the police station; call 112 if you are in danger, 181 Women Helpline.",
            ],
            "documents": [
                "Screenshots of the profile, posts, messages with date and time",
                "Profile links and usernames",
                "Your ID proof",
            ],
            "tell": [
                "Who you suspect, if anyone, and since when it is happening",
                "What exactly was posted or sent",
                "Which platform and the profile link",
            ],
            "helplines": HELP_EN + ["181 Women Helpline", "1098 Childline"],
        },
        "hi": {
            "title": "ऑनलाइन उत्पीड़न या फर्जी प्रोफ़ाइल",
            "where": [
                "प्रोफ़ाइल लिंक, यूज़रनेम, तारीख और समय के साथ स्क्रीनशॉट लें। चैट डिलीट न करें।",
                "प्लेटफ़ॉर्म पर प्रोफ़ाइल रिपोर्ट और ब्लॉक करें।",
                "cybercrime.gov.in पर शिकायत करें (महिला और बाल संबंधी अपराध गुमनाम रूप से भी रिपोर्ट हो सकते हैं)।",
                "थाने में लिखित शिकायत दें; खतरा हो तो 112, महिला हेल्पलाइन 181।",
            ],
            "documents": [
                "प्रोफ़ाइल, पोस्ट और मैसेज के तारीख-समय सहित स्क्रीनशॉट",
                "प्रोफ़ाइल लिंक और यूज़रनेम",
                "आपका पहचान पत्र",
            ],
            "tell": [
                "किस पर शक है (यदि है) और कब से हो रहा है",
                "ठीक क्या पोस्ट या भेजा गया",
                "कौन सा प्लेटफ़ॉर्म और प्रोफ़ाइल लिंक",
            ],
            "helplines": HELP_HI + ["181 महिला हेल्पलाइन", "1098 चाइल्डलाइन"],
        },
    },
    "sextortion_or_morphed_images": {
        "en": {
            "title": "Sextortion, blackmail or morphed / intimate images",
            "where": [
                "Do not pay and do not delete the chat. Take screenshots of the profile, threats and payment demands.",
                "Block and report the profile on the platform.",
                "File a complaint on cybercrime.gov.in; you can report anonymously in the women and children related section.",
                "Use StopNCII.org to help block intimate images on participating platforms.",
                "Call 1930 if you already paid; call 112 or 181 if you feel unsafe.",
            ],
            "documents": [
                "Screenshots of threats, profile link, phone number, payment details",
                "Chat history (do not delete)",
                "Your ID proof",
            ],
            "tell": [
                "When and on which platform it started",
                "What was demanded and whether you paid (amount, UPI/bank details)",
                "What images or videos are being threatened",
            ],
            "helplines": HELP_EN + ["181 Women Helpline", "1098 Childline"],
        },
        "hi": {
            "title": "सेक्सटॉर्शन, ब्लैकमेल या मॉर्फ्ड / निजी तस्वीरें",
            "where": [
                "पैसे न दें और चैट डिलीट न करें। प्रोफ़ाइल, धमकियों और पैसे की मांग के स्क्रीनशॉट लें।",
                "प्रोफ़ाइल को ब्लॉक और रिपोर्ट करें।",
                "cybercrime.gov.in पर शिकायत करें; महिला और बाल संबंधी सेक्शन में गुमनाम रिपोर्ट कर सकते हैं।",
                "निजी तस्वीरें रोकने में मदद के लिए StopNCII.org का उपयोग करें।",
                "पैसे दे दिए हों तो 1930 पर कॉल करें; असुरक्षित लगे तो 112 या 181।",
            ],
            "documents": [
                "धमकियों, प्रोफ़ाइल लिंक, फोन नंबर और पेमेंट डिटेल के स्क्रीनशॉट",
                "चैट हिस्ट्री (डिलीट न करें)",
                "आपका पहचान पत्र",
            ],
            "tell": [
                "कब और किस प्लेटफ़ॉर्म पर शुरू हुआ",
                "क्या मांगा गया और क्या आपने पैसे दिए (राशि, UPI/बैंक विवरण)",
                "किन फोटो या वीडियो की धमकी दी जा रही है",
            ],
            "helplines": HELP_HI + ["181 महिला हेल्पलाइन", "1098 चाइल्डलाइन"],
        },
    },
    "lost_phone_or_sim_fraud": {
        "en": {
            "title": "Lost or stolen phone, SIM misuse, suspicious calls",
            "where": [
                "Lost phone: give a police complaint, then block the phone on sancharsaathi.gov.in using the IMEI and the complaint copy.",
                "Ask your bank to block UPI and cards linked to that number; ask your operator to block the SIM.",
                "Check mobile connections in your name on sancharsaathi.gov.in and report numbers you do not use.",
                "Report a suspicious call, SMS or WhatsApp number on Chakshu at sancharsaathi.gov.in.",
            ],
            "documents": [
                "Phone IMEI number (box, bill, or dial star hash 06 hash on another phone with the same account)",
                "Mobile number and ID proof",
                "Copy of the police complaint or FIR",
                "Screenshots of suspicious messages or call log",
            ],
            "tell": [
                "Make, model, IMEI and mobile number",
                "Date, time and place the phone was lost or stolen",
                "Details of the suspicious call or SMS",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "खोया / चोरी हुआ फोन, सिम का दुरुपयोग, संदिग्ध कॉल",
            "where": [
                "फोन खोने पर पुलिस में शिकायत दें, फिर IMEI और शिकायत की कॉपी के साथ sancharsaathi.gov.in पर फोन ब्लॉक करें।",
                "बैंक से उस नंबर से जुड़े UPI और कार्ड ब्लॉक करवाएं; ऑपरेटर से सिम ब्लॉक करवाएं।",
                "sancharsaathi.gov.in पर अपने नाम के मोबाइल कनेक्शन जांचें और जो आप इस्तेमाल नहीं करते उन्हें रिपोर्ट करें।",
                "संदिग्ध कॉल, SMS या WhatsApp नंबर की sancharsaathi.gov.in पर Chakshu में रिपोर्ट करें।",
            ],
            "documents": [
                "फोन का IMEI नंबर (डिब्बा, बिल, या उसी अकाउंट वाले दूसरे फोन पर star hash 06 hash डायल करें)",
                "मोबाइल नंबर और पहचान पत्र",
                "पुलिस शिकायत या FIR की कॉपी",
                "संदिग्ध मैसेज या कॉल लॉग के स्क्रीनशॉट",
            ],
            "tell": [
                "फोन का मेक, मॉडल, IMEI और मोबाइल नंबर",
                "फोन कब, कहाँ खोया या चोरी हुआ",
                "संदिग्ध कॉल या SMS का विवरण",
            ],
            "helplines": HELP_HI,
        },
    },
    "job_loan_investment_scam": {
        "en": {
            "title": "Job, loan app, investment or trading scam",
            "where": [
                "Stop paying anyone. Do not pay any 'fee to withdraw'.",
                "Call 1930 immediately if you transferred money and note the acknowledgement number.",
                "File the complaint on cybercrime.gov.in with the app, website and payment details.",
                "Give a written complaint at the police station; tell your bank to hold the beneficiary account.",
            ],
            "documents": [
                "Chats, offer letters, group and profile links, app name and website",
                "Payment receipts, bank statement, UTR numbers, beneficiary account details",
                "Screenshots of fake profit or threats (loan app blackmail)",
            ],
            "tell": [
                "How you were contacted and by whom",
                "Every payment: date, amount, account or UPI ID",
                "What was promised and what happened when you tried to withdraw",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "नौकरी, लोन ऐप, निवेश या ट्रेडिंग ठगी",
            "where": [
                "किसी को और पैसे न दें। 'निकासी शुल्क' बिल्कुल न दें।",
                "पैसे भेजे हों तो तुरंत 1930 पर कॉल करें और acknowledgement नंबर नोट करें।",
                "ऐप, वेबसाइट और पेमेंट विवरण के साथ cybercrime.gov.in पर शिकायत करें।",
                "थाने में लिखित शिकायत दें; बैंक से लाभार्थी खाता होल्ड करवाने को कहें।",
            ],
            "documents": [
                "चैट, ऑफर लेटर, ग्रुप और प्रोफ़ाइल लिंक, ऐप का नाम और वेबसाइट",
                "पेमेंट रसीद, बैंक स्टेटमेंट, UTR नंबर, लाभार्थी खाता विवरण",
                "नकली मुनाफे या धमकियों (लोन ऐप ब्लैकमेल) के स्क्रीनशॉट",
            ],
            "tell": [
                "आपसे किसने और कैसे संपर्क किया",
                "हर भुगतान: तारीख, राशि, खाता या UPI ID",
                "क्या वादा किया गया और निकासी की कोशिश पर क्या हुआ",
            ],
            "helplines": HELP_HI,
        },
    },
    "suspicious_call_or_sms": {
        "en": {
            "title": "Suspicious or fake call, SMS or WhatsApp message (no money lost yet)",
            "where": [
                "Do not click any link, install any app or share any OTP/PIN. Block the number.",
                "Report the number or message on Chakshu at sancharsaathi.gov.in (report suspected fraud communication).",
                "If you already shared an OTP/PIN, clicked a link, installed an app or paid, call 1930 immediately and tell your bank.",
                "You can also file a complaint on cybercrime.gov.in.",
            ],
            "documents": [
                "Screenshots of the SMS or WhatsApp message and any link",
                "The caller's number and the call log with date and time",
                "Details of any app installed or payment made",
            ],
            "tell": [
                "The phone number or sender ID, date and time",
                "What the caller or message said or asked you to do",
                "Whether you clicked, shared an OTP or paid anything",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "संदिग्ध या फर्जी कॉल, SMS या WhatsApp मैसेज (अभी पैसे नहीं गए)",
            "where": [
                "कोई लिंक न खोलें, कोई ऐप इंस्टॉल न करें और OTP/PIN न बताएं। नंबर ब्लॉक करें।",
                "sancharsaathi.gov.in पर Chakshu में नंबर या मैसेज रिपोर्ट करें (suspected fraud communication)।",
                "अगर OTP/PIN बता दिया, लिंक खोल लिया, ऐप इंस्टॉल कर लिया या पैसे दे दिए, तो तुरंत 1930 पर कॉल करें और बैंक को बताएं।",
                "आप cybercrime.gov.in पर भी शिकायत कर सकते हैं।",
            ],
            "documents": [
                "SMS या WhatsApp मैसेज और लिंक के स्क्रीनशॉट",
                "कॉल करने वाले का नंबर और तारीख-समय सहित कॉल लॉग",
                "इंस्टॉल किए ऐप या किए गए भुगतान का विवरण",
            ],
            "tell": [
                "फोन नंबर या sender ID, तारीख और समय",
                "कॉल या मैसेज में क्या कहा गया या क्या करने को कहा गया",
                "क्या आपने लिंक खोला, OTP बताया या कुछ भुगतान किया",
            ],
            "helplines": HELP_HI,
        },
    },
    "tenant_verification": {
        "en": {
            "title": "Tenant or domestic servant verification (application)",
            "where": [
                "Use the Rajasthan Police citizen services (police.rajasthan.gov.in / RajCop app) for tenant and servant verification, or go to your local police station.",
                "Fill the form with the owner and tenant/servant details, attach documents and photographs, submit and keep the receipt.",
                "Confirm the current process with your local police station.",
            ],
            "documents": [
                "Copy of Aadhaar Card or Voter ID of the tenant/servant",
                "Two passport size photographs",
                "Permanent address and phone number of the tenant/servant",
                "Previous landlord's or employer's address and phone number",
                "Owner's ID proof and the property address",
            ],
            "tell": [
                "Owner's name, address and phone",
                "Tenant/servant's full name, permanent address, phone",
                "Date of joining or start of tenancy",
            ],
            "helplines": ["112 Police Emergency", "8764874306 Alwar Cyber Control Room"],
        },
        "hi": {
            "title": "किरायेदार / घरेलू नौकर सत्यापन (आवेदन)",
            "where": [
                "किरायेदार और नौकर सत्यापन के लिए राजस्थान पुलिस की नागरिक सेवाएं (police.rajasthan.gov.in / RajCop ऐप) इस्तेमाल करें, या अपने स्थानीय थाने जाएं।",
                "मकान मालिक और किरायेदार/नौकर का विवरण भरें, दस्तावेज़ और फोटो लगाएं, जमा करें और रसीद रखें।",
                "वर्तमान प्रक्रिया की पुष्टि अपने स्थानीय थाने से करें।",
            ],
            "documents": [
                "किरायेदार/नौकर के आधार कार्ड या वोटर ID की प्रति",
                "2 पासपोर्ट साइज फोटो",
                "किरायेदार/नौकर का स्थायी पता और फोन नंबर",
                "पिछले मकान मालिक या नियोक्ता का पता और फोन नंबर",
                "मालिक का पहचान पत्र और संपत्ति का पता",
            ],
            "tell": [
                "मालिक का नाम, पता और फोन",
                "किरायेदार/नौकर का पूरा नाम, स्थायी पता, फोन",
                "काम या किरायेदारी शुरू होने की तारीख",
            ],
            "helplines": ["112 पुलिस आपातकाल", "8764874306 अलवर साइबर कंट्रोल रूम"],
        },
    },
    "other_cyber_crime": {
        "en": {
            "title": "Any other cyber crime",
            "where": [
                "Save all evidence first: screenshots, links, chats, call log. Do not delete anything.",
                "File the complaint on cybercrime.gov.in.",
                "Give a written complaint at the nearest police station or to the Alwar Cyber Control Room (8764874306); ask for a receipt.",
                "If money was lost, call 1930 immediately.",
            ],
            "documents": [
                "Screenshots, links, chats, emails, call log",
                "Details of the suspect: phone, email, profile, website, account",
                "Your ID proof",
            ],
            "tell": [
                "What happened, in time order, with dates",
                "Who or what is involved and how you came in contact",
                "Any loss: money, data, account, reputation",
            ],
            "helplines": HELP_EN,
        },
        "hi": {
            "title": "कोई अन्य साइबर अपराध",
            "where": [
                "पहले सारे सबूत सुरक्षित करें: स्क्रीनशॉट, लिंक, चैट, कॉल लॉग। कुछ भी डिलीट न करें।",
                "cybercrime.gov.in पर शिकायत दर्ज करें।",
                "नजदीकी थाने या अलवर साइबर कंट्रोल रूम (8764874306) पर लिखित शिकायत दें; रसीद लें।",
                "पैसे का नुकसान हुआ हो तो तुरंत 1930 पर कॉल करें।",
            ],
            "documents": [
                "स्क्रीनशॉट, लिंक, चैट, ईमेल, कॉल लॉग",
                "संदिग्ध का विवरण: फोन, ईमेल, प्रोफ़ाइल, वेबसाइट, खाता",
                "आपका पहचान पत्र",
            ],
            "tell": [
                "क्या हुआ, समय के क्रम में, तारीखों सहित",
                "कौन या क्या शामिल है और संपर्क कैसे हुआ",
                "क्या नुकसान हुआ: पैसे, डेटा, अकाउंट, प्रतिष्ठा",
            ],
            "helplines": HELP_HI,
        },
    },
}


def render_guide(incident_type: str, lang: str) -> tuple[str, str] | None:
    """Return (title, plain-text checklist) or None for an unknown incident_type."""
    guide = GUIDES.get(incident_type, {}).get(lang)
    if guide is None:
        return None
    labels = LABELS[lang]
    lines = [guide["title"], ""]
    for key in ("where", "documents", "tell", "helplines"):
        lines.append(labels[key])
        marker = "{n}. " if key == "where" else "• "
        lines += [marker.format(n=i) + item for i, item in enumerate(guide[key], 1)]
        lines.append("")
    return guide["title"], "\n".join(lines).strip()
