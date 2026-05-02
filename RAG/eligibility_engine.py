from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================

MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "niti_setu"
COLLECTION_NAME = "scheme_chunks"
VECTOR_INDEX_NAME = "vector_index"

if not MONGO_URI:
    raise Exception("MONGO_URI not set")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ==============================
# SPECIAL SUBSIDY STATES
# ==============================

SPECIAL_SUBSIDY_STATES = [
    "Assam","Meghalaya","Manipur","Nagaland",
    "Tripura","Mizoram","Arunachal Pradesh",
    "Sikkim","Jammu & Kashmir","Himachal Pradesh",
    "Uttarakhand","Lakshadweep","Andaman & Nicobar"
]


# ==============================
# RAG RETRIEVAL
# ==============================

def retrieve_proof(query, scheme_name, limit=4):

    try:

        embedding = model.encode(query).tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 300,
                    "limit": limit
                }
            },
            {"$match": {"scheme": scheme_name}},
            {
                "$project": {
                    "_id": 0,
                    "page": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        formatted = []
        seen = set()

        for r in results:

            snippet = r["text"].strip()

            key = snippet[:120]

            if key in seen:
                continue

            seen.add(key)

            formatted.append({
                "document_name": f"{scheme_name} Official Guidelines",
                "page_number": r["page"],
                "section": "Official Scheme Documentation",
                "line_reference": f"Page {r['page']} excerpt",
                "snippet": snippet,
                "relevance_score": round(r["score"],3)
            })

        return formatted

    except Exception:
        return []


# ==============================
# PM-KISAN LOGIC
# ==============================

def evaluate_pm_kisan(user):

    if user["land_size"] <= 0:
        return False,"PMK_NO_LAND","No cultivable land.",0

    if user["income_tax_payer"]:
        return False,"PMK_TAX_EXCLUDED","Income tax payer excluded.",0

    if user["pension"] > 10000:
        return False,"PMK_HIGH_PENSION","Pension above ₹10,000 excluded.",0

    score = 60

    if user["land_size"] <= 2:
        score += 20

    if user["annual_income"] > 800000:
        score -= 20

    if user["annual_income"] > 2000000:
        score -= 40

    score = max(score,0)

    return True,"PMK_ELIGIBLE","Eligible landholding farmer family.",score


# ==============================
# PM-KUSUM LOGIC
# ==============================

def evaluate_pm_kusum(user):

    land = user["land_size"]
    electricity = user["electricity_connection"]
    crop = user["crop_type"]
    category = user["category"]
    state = user["state"]

    subsidy = 60

    if category in ["SC","ST"]:
        subsidy = 90

    if state in SPECIAL_SUBSIDY_STATES:
        subsidy = max(subsidy,70)

    water_intensive = ["Paddy","Sugarcane"]

    components = {}
    best_component = None
    best_score = 0

    eligible_a = land >= 2
    score_a = 70 if eligible_a else 0

    components["Component_A"]={
        "eligible":eligible_a,
        "reason":"Requires minimum 2 acres.",
        "subsidy_percent":subsidy if eligible_a else None,
        "component_score":score_a
    }

    eligible_b = land > 0 and not electricity
    score_b = 85 if eligible_b else 0

    if eligible_b and crop in water_intensive:
        score_b += 10

    components["Component_B"]={
        "eligible":eligible_b,
        "reason":"Standalone solar pump for non-grid farmers.",
        "subsidy_percent":subsidy if eligible_b else None,
        "component_score":score_b
    }

    eligible_c = land > 0 and electricity
    score_c = 75 if eligible_c else 0

    components["Component_C"]={
        "eligible":eligible_c,
        "reason":"Solarization of grid-connected pump.",
        "subsidy_percent":subsidy if eligible_c else None,
        "component_score":score_c
    }

    for name,data in components.items():

        if data["eligible"] and data["component_score"] > best_score:
            best_score = data["component_score"]
            best_component = name

    return components,best_score,best_component


# ==============================
# AIF LOGIC
# ==============================

def evaluate_aif(user):

    if user["land_size"] <= 0:
        return False,"AIF_NO_BASE","Agricultural base required.",0

    if user["annual_income"] > 5000000:
        return False,"AIF_LOW_PRIORITY","Very high income reduces priority.",0

    score = 40

    if user["category"] in ["SC","ST"]:
        score += 20

    if user["crop_type"] in ["Fruits","Vegetables","Horticulture"]:
        score += 20

    score += 20

    return True,"AIF_ELIGIBLE","Eligible for AIF financing support.",score


# ==============================
# IMPACT SCORING
# ==============================

def calculate_impact_score(name,base_score):

    financial = 1.0
    infra = 1.0

    if name == "PM-KISAN":
        financial = 1.2
        infra = 0.3

    if "PM-KUSUM" in name:
        financial = 1.0
        infra = 1.5

    if "Agriculture Infrastructure Fund" in name:
        financial = 1.3
        infra = 1.4

    return base_score * (financial + infra)


# ==============================
# MAIN ENGINE
# ==============================

def eligibility_engine(user):

    results={}
    scores={}

    # PM-KISAN

    k_ok,k_code,k_msg,k_score = evaluate_pm_kisan(user)

    k_proof = retrieve_proof(
        "PM-KISAN eligibility income tax pension rules",
        "PM-KISAN"
    )

    confidence_kisan = min((k_score/100)+(len(k_proof)*0.05),1.0)

    results["PM-KISAN"]={
        "eligible":k_ok,
        "reason_code":k_code,
        "reason_message":k_msg,
        "benefit_summary":"₹6,000 per year income support" if k_ok else None,
        "confidence_score":round(confidence_kisan,2),
        "proof":k_proof,
        "next_steps":[
            "Verify land records",
            "Link Aadhaar to bank",
            "Check PM-KISAN portal"
        ]
    }

    if k_ok:
        scores["PM-KISAN"]=calculate_impact_score("PM-KISAN",k_score)


    # PM-KUSUM

    components,kusum_score,best_component = evaluate_pm_kusum(user)

    kusum_proof = retrieve_proof(
        f"PM-KUSUM {best_component} subsidy eligibility",
        "PM-KUSUM"
    )

    confidence_kusum = min((kusum_score/100)+(len(kusum_proof)*0.05),1.0)

    results["PM-KUSUM"]={
        "eligible":best_component is not None,
        "best_component":best_component,
        "confidence_score":round(confidence_kusum,2),
        "components":components,
        "proof":kusum_proof,
        "next_steps":[
            "Visit renewable energy office",
            "Apply for subsidy",
            "Submit electricity and land proof"
        ]
    }

    if best_component:
        scores[f"PM-KUSUM - {best_component}"] = calculate_impact_score(
            "PM-KUSUM",
            kusum_score
        )


    # AIF

    a_ok,a_code,a_msg,a_score = evaluate_aif(user)

    a_proof = retrieve_proof(
        "Agriculture Infrastructure Fund interest subvention eligibility",
        "AIF"
    )

    confidence_aif = min((a_score/100)+(len(a_proof)*0.05),1.0)

    results["Agriculture Infrastructure Fund (AIF)"]={
        "eligible":a_ok,
        "reason_code":a_code,
        "reason_message":a_msg,
        "benefit_summary":"3% interest subvention up to ₹2 crore for 7 years",
        "confidence_score":round(confidence_aif,2),
        "proof":a_proof,
        "next_steps":[
            "Prepare project report",
            "Approach bank",
            "Apply through AIF portal"
        ]
    }

    if a_ok:
        scores["Agriculture Infrastructure Fund (AIF)"] = calculate_impact_score(
            "Agriculture Infrastructure Fund",
            a_score
        )


    # =============================
    # RECOMMENDATION
    # =============================

    recommended = max(scores,key=scores.get) if scores else "None"

    results["scheme_comparison"]={
        "total_eligible_schemes":len(scores),
        "eligible_schemes":list(scores.keys()),
        "recommended_scheme":recommended,
        "comparison_note":
        f"You qualify for {len(scores)} schemes. {recommended} provides highest overall impact."
        if scores else
        "No schemes matched."
    }


    # =============================
    # EXPLAINABLE AI BLOCK
    # =============================

    results["decision_explanation"]={
        "key_factors":[
            f"Land size: {user['land_size']} acres",
            f"Annual income: ₹{user['annual_income']}",
            f"Electricity connection: {user['electricity_connection']}",
            f"Category: {user['category']}",
            f"Income tax payer: {user['income_tax_payer']}"
        ],
        "decision_reason":
        f"{recommended} recommended because it provides the best balance of eligibility score, financial benefit, and infrastructure impact."
        if recommended!="None"
        else "No scheme satisfied eligibility conditions."
    }

    results["summary"] = (
        f"Recommended scheme: {recommended}"
        if recommended!="None"
        else "Currently not eligible for these schemes."
    )

    return results