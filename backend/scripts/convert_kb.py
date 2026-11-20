"""Script to convert kb/*.yaml knowledge files into structured, domain-categorized Markdown
documents in backend/kb/ and sync raw YAML into backend/kb/yaml/.
"""

import os
import shutil
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_KB_DIR = ROOT_DIR / "kb"
DEST_KB_DIR = ROOT_DIR / "backend" / "kb"

CATEGORY_TO_SUBDIR = {
    "financial_fraud": "financial-fraud",
    "portals_reporting": "portals-reporting",
    "device_malware": "device-malware",
    "social_media": "social-media",
    "scam_alerts": "scam-alerts",
    "cyber_hygiene": "cyber-hygiene",
}

CATEGORY_TO_DOMAIN = {
    "financial_fraud": "financial_fraud",
    "portals_reporting": "reporting",
    "device_malware": "malware",
    "social_media": "social_media",
    "scam_alerts": "financial_fraud",
    "cyber_hygiene": "cyber_safety",
}

# Fine-grained domain overrides based on specific scam type
ID_DOMAIN_OVERRIDES = {
    # portals / reporting
    "mrm_portal_recovery": "reporting",
    "grm_portal_grievance": "reporting",
    "evidence_checklist": "evidence",
    "thana_finder_help": "local_police",
    "ncrp_vs_fir": "reporting",
    "bank_account_freeze": "reporting",
    # device / malware / telecom
    "call_forwarding_scam": "telecom",
    "phone_lost_stolen": "device",
    "sim_swap_fraud": "telecom",
    "apk_malware_cleanup": "malware",
    "remote_access_scam": "malware",
    "challan_parcel_scam": "malware",
    # social media & harassment
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
    # phishing / scam alerts
    "spot_phishing_quiz": "cyber_safety",
    "govt_subsidy_tax_scam": "phishing",
}


def yaml_to_markdown(data: dict, filename: str) -> str:
    fid = data.get("id", Path(filename).stem)
    cat = data.get("category", "general")
    urgency = data.get("urgency", "medium")
    title_en = data.get("title_en", "")
    title_hi = data.get("title_hi", "")
    summary_en = data.get("summary_en", "")
    summary_hi = data.get("summary_hi", "")
    source = data.get("source", "official_government")
    last_rev = data.get("last_reviewed", "2026-11-15")

    domain = ID_DOMAIN_OVERRIDES.get(fid, CATEGORY_TO_DOMAIN.get(cat, "general"))

    lines = [
        "---",
        f"doc_id: {fid}",
        f"category: {cat}",
        f"domain: {domain}",
        f"subtopic: {fid}",
        f"urgency: {urgency}",
        "jurisdiction: national",
        f"authority: {source}",
        "language: bilingual",
        f"last_verified: {last_rev}",
        "---",
        f"# {title_en} / {title_hi}",
        "",
        "## Summary & Overview",
        f"**English:** {summary_en}",
        "",
        f"**हिंदी:** {summary_hi}",
        "",
        "## Immediate Action Steps (English)",
    ]
    for step in data.get("answer_en", []):
        lines.append(f"- {step}")
    lines.append("")

    lines.append("## तत्काल कदम और समाधान (Hindi Action Guide)")
    for step in data.get("answer_hi", []):
        lines.append(f"- {step}")
    lines.append("")

    lines.append("## Queries, Keywords & Variations")
    kw_en = data.get("keywords_en", [])
    if kw_en:
        lines.append(f"- **Keywords (EN):** {', '.join(kw_en)}")
    kw_hi = data.get("keywords_hi", [])
    if kw_hi:
        lines.append(f"- **Keywords (HI):** {', '.join(kw_hi)}")
    kw_hing = data.get("keywords_hinglish", [])
    if kw_hing:
        lines.append(f"- **Keywords (Hinglish):** {', '.join(kw_hing)}")
    samples = data.get("sample_queries", [])
    if samples:
        lines.append(f"- **Sample Queries:** {'; '.join(samples)}")
    lines.append("")

    links = data.get("official_links", [])
    if links:
        lines.append("## Official Helplines & Links")
        for l in links:
            name_en = l.get("name_en", "")
            name_hi = l.get("name_hi", "")
            url = l.get("url", "")
            verified = "Verified" if l.get("verified") else "Official"
            lines.append(f"- [{name_en} ({name_hi})]({url}) — {verified}")
        lines.append("")

    followups = data.get("followups", [])
    if followups:
        lines.append("## Follow-up Guidance & Next Steps")
        for fo in followups:
            label_en = fo.get("label_en", "")
            label_hi = fo.get("label_hi", "")
            intent = fo.get("intent_id", "")
            lines.append(f"- **{label_en}** ({label_hi}) [Ref: {intent}]")
        lines.append("")

    return "\n".join(lines)


import json

SCRATCH_DIR = Path(r"C:\Users\daayu\.gemini\antigravity-cli\brain\bdb4481f-0446-4534-a826-407cff303534\scratch")


def main():
    if not SRC_KB_DIR.exists():
        print(f"Source kb dir not found: {SRC_KB_DIR}")
        return

    yaml_files = sorted(SRC_KB_DIR.glob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files in {SRC_KB_DIR}")

    out_bundle = {}
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            raw_text = f.read()
            data = yaml.safe_load(raw_text)

        cat = data.get("category", "general")
        sub_dir_name = CATEGORY_TO_SUBDIR.get(cat, "general")
        md_content = yaml_to_markdown(data, yf.name)
        
        md_rel_path = f"backend/kb/{sub_dir_name}/{yf.stem}.md"
        yaml_rel_path = f"backend/kb/yaml/{yf.name}"
        
        out_bundle[md_rel_path] = md_content
        out_bundle[yaml_rel_path] = raw_text

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SCRATCH_DIR / "converted_kb.json"
    with open(out_json, "w", encoding="utf-8") as out:
        json.dump(out_bundle, out, ensure_ascii=False, indent=2)

    print(f"Successfully packaged {len(out_bundle)} files into {out_json}")


if __name__ == "__main__":
    main()
