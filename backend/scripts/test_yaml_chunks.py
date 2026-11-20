import glob
import yaml
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    domain: str = "general"
    subtopic: str = ""
    jurisdiction: str = "national"
    authority: str = "official_government"

CATEGORY_TO_DOMAIN = {
    "financial_fraud": "financial_fraud",
    "portals_reporting": "reporting",
    "device_malware": "malware",
    "social_media": "social_media",
    "scam_alerts": "financial_fraud",
    "cyber_hygiene": "cyber_safety",
}

ID_DOMAIN_OVERRIDES = {
    "mrm_portal_recovery": "reporting",
    "grm_portal_grievance": "reporting",
    "evidence_checklist": "evidence",
    "thana_finder_help": "local_police",
    "ncrp_vs_fir": "reporting",
    "bank_account_freeze": "reporting",
    "call_forwarding_scam": "telecom",
    "phone_lost_stolen": "device",
    "sim_swap_fraud": "telecom",
    "apk_malware_cleanup": "malware",
    "remote_access_scam": "malware",
    "challan_parcel_scam": "malware",
    "whatsapp_hacked": "account_hacking",
    "facebook_hacked": "account_hacking",
    "instagram_hacked": "account_hacking",
    "gmail_google_hacked": "account_hacking",
    "telegram_hacked": "account_hacking",
    "x_twitter_hacked": "account_hacking",
    "morphed_photos_stopncii": "deepfake",
    "deepfake_voice_scam": "deepfake",
    "sextortion_blackmail": "sextortion",
    "cyber_harassment_stalking": "harassment",
    "loan_app_harassment": "harassment",
    "spot_phishing_quiz": "cyber_safety",
    "govt_subsidy_tax_scam": "phishing",
}

def chunk_text(text, size=900):
    pieces, buf = [], ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) > size:
            pieces.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        pieces.append(buf.strip())
    return pieces

def parse_yaml_chunks(data, filename, size=900):
    fid = data.get("id") or filename.replace(".yaml", "").replace(".yml", "")
    cat = data.get("category", "general")
    domain = ID_DOMAIN_OVERRIDES.get(fid, CATEGORY_TO_DOMAIN.get(cat, "general"))
    subtopic = data.get("subtopic") or fid
    jurisdiction = data.get("jurisdiction", "national")
    authority = data.get("source", "official_government")
    title_en = data.get("title_en", "")
    title_hi = data.get("title_hi", "")
    full_title = f"{title_en} / {title_hi}" if (title_en and title_hi) else (title_en or title_hi or fid)

    chunks = []
    
    # 1. Summary
    sum_en = data.get("summary_en", "")
    sum_hi = data.get("summary_hi", "")
    if sum_en or sum_hi:
        sum_body = f"Summary:\nEnglish: {sum_en}\nHindi: {sum_hi}"
        chunks.append(Chunk(
            id=f"{fid}-summary-0",
            text=f"{full_title} > Summary\n{sum_body}",
            source=filename,
            section="Summary",
            domain=domain,
            subtopic=subtopic,
            jurisdiction=jurisdiction,
            authority=authority,
        ))

    # 2. English Action Steps
    ans_en = data.get("answer_en", [])
    if ans_en:
        steps_text = "\n".join(f"{i+1}. {s}" if not s.strip().startswith(tuple("0123456789")) else s for i, s in enumerate(ans_en))
        for p_idx, piece in enumerate(chunk_text(steps_text, size)):
            chunks.append(Chunk(
                id=f"{fid}-en-steps-{p_idx}",
                text=f"{title_en} > Immediate Action & Steps (English)\n{piece}",
                source=filename,
                section="Immediate Action & Steps (English)",
                domain=domain,
                subtopic=subtopic,
                jurisdiction=jurisdiction,
                authority=authority,
            ))

    # 3. Hindi Action Steps
    ans_hi = data.get("answer_hi", [])
    if ans_hi:
        steps_text = "\n".join(f"{i+1}. {s}" if not s.strip().startswith(tuple("0123456789")) else s for i, s in enumerate(ans_hi))
        for p_idx, piece in enumerate(chunk_text(steps_text, size)):
            chunks.append(Chunk(
                id=f"{fid}-hi-steps-{p_idx}",
                text=f"{title_hi} > तत्काल कदम और समाधान (Hindi Guide)\n{piece}",
                source=filename,
                section="तत्काल कदम और समाधान (Hindi Guide)",
                domain=domain,
                subtopic=subtopic,
                jurisdiction=jurisdiction,
                authority=authority,
            ))

    # 4. Queries & Keywords
    kw_en = data.get("keywords_en", [])
    kw_hi = data.get("keywords_hi", [])
    kw_hing = data.get("keywords_hinglish", [])
    samples = data.get("sample_queries", [])
    if kw_en or kw_hi or kw_hing or samples:
        parts = []
        if kw_en: parts.append("Keywords (EN): " + ", ".join(kw_en))
        if kw_hi: parts.append("Keywords (HI): " + ", ".join(kw_hi))
        if kw_hing: parts.append("Keywords (Hinglish): " + ", ".join(kw_hing))
        if samples: parts.append("Common Queries: " + "; ".join(samples))
        chunks.append(Chunk(
            id=f"{fid}-keywords-0",
            text=f"{full_title} > Queries & Keywords\n" + "\n".join(parts),
            source=filename,
            section="Queries & Keywords",
            domain=domain,
            subtopic=subtopic,
            jurisdiction=jurisdiction,
            authority=authority,
        ))

    # 5. Official Links & Helplines
    links = data.get("official_links", [])
    if links:
        link_lines = [f"- {l.get('name_en', '')} ({l.get('name_hi', '')}): {l.get('url', '')}" for l in links]
        chunks.append(Chunk(
            id=f"{fid}-links-0",
            text=f"{full_title} > Official Helplines & Links\n" + "\n".join(link_lines),
            source=filename,
            section="Official Helplines & Links",
            domain=domain,
            subtopic=subtopic,
            jurisdiction=jurisdiction,
            authority=authority,
        ))

    return chunks

if __name__ == "__main__":
    files = sorted(glob.glob("kb/*.yaml"))
    all_chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            d = yaml.safe_load(fp)
        c = parse_yaml_chunks(d, f)
        all_chunks.extend(c)

    print(f"Total YAML files parsed: {len(files)}")
    print(f"Total Chunks produced: {len(all_chunks)}")
    domains = {}
    for c in all_chunks:
        domains[c.domain] = domains.get(c.domain, 0) + 1
    print("Domain distribution:")
    for d, cnt in sorted(domains.items()):
        print(f"  {d}: {cnt} chunks")
