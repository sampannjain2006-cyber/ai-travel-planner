from flask import Flask, render_template, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
from ibmcloudant.cloudant_v1 import CloudantV1, Document
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator as CloudantIAMAuthenticator
import json
import subprocess
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException

load_dotenv()

app = Flask(__name__)

# ── IBM NLU setup ──────────────────────────────────────────────────────────────
nlu_apikey = os.environ.get("NLU_API_KEY")
nlu_url = os.environ.get("NLU_URL")
nlu_authenticator = IAMAuthenticator(nlu_apikey)
nlu = NaturalLanguageUnderstandingV1(version='2021-08-01', authenticator=nlu_authenticator)
nlu.set_service_url(nlu_url)

# ── Gemini API setup ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# ── IBM Cloudant setup ────────────────────────────────────────────────────────
CLOUDANT_API_KEY = os.environ.get("CLOUDANT_API_KEY")
CLOUDANT_URL = os.environ.get("CLOUDANT_URL")
CLOUDANT_DB_NAME = "travel_history"

cloudant_authenticator = CloudantIAMAuthenticator(CLOUDANT_API_KEY)
cloudant_client = CloudantV1(authenticator=cloudant_authenticator)
cloudant_client.set_service_url(CLOUDANT_URL)


def ensure_database_exists():
    """Create the travel_history database if it doesn't exist."""
    try:
        db_list = cloudant_client.get_all_dbs().get_result()
        if CLOUDANT_DB_NAME not in db_list:
            cloudant_client.put_database(db=CLOUDANT_DB_NAME).get_result()
            print(f"Created Cloudant database: {CLOUDANT_DB_NAME}")
        else:
            print(f"Cloudant database already exists: {CLOUDANT_DB_NAME}")
    except Exception as e:
        print(f"Cloudant DB init error: {e}")


# Initialize DB on startup
ensure_database_exists()


def save_to_cloudant(query, location, days, budget, plan):
    """Save a search result to Cloudant."""
    try:
        doc = Document(
            query=query,
            location=location or "Unknown",
            days=days or 0,
            budget=budget or 0,
            plan=plan,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        response = cloudant_client.post_document(
            db=CLOUDANT_DB_NAME,
            document=doc
        ).get_result()
        print(f"Saved to Cloudant: {response['id']}")
        return response['id']
    except Exception as e:
        print(f"Cloudant save error: {e}")
        return None


def get_history_from_cloudant():
    """Fetch all search history from Cloudant, sorted by newest first."""
    try:
        response = cloudant_client.post_all_docs(
            db=CLOUDANT_DB_NAME,
            include_docs=True
        ).get_result()
        docs = []
        for row in response.get('rows', []):
            doc = row.get('doc', {})
            if not doc.get('_id', '').startswith('_design'):
                docs.append({
                    'id': doc.get('_id'),
                    'query': doc.get('query', ''),
                    'location': doc.get('location', 'Unknown'),
                    'days': doc.get('days', 0),
                    'budget': doc.get('budget', 0),
                    'plan': doc.get('plan', []),
                    'timestamp': doc.get('timestamp', '')
                })
        # Sort by timestamp descending (newest first)
        docs.sort(key=lambda x: x['timestamp'], reverse=True)
        return docs
    except Exception as e:
        print(f"Cloudant fetch error: {e}")
        return []


def delete_from_cloudant(doc_id):
    """Delete a single history entry from Cloudant."""
    try:
        # First get the document to get its revision
        doc = cloudant_client.get_document(
            db=CLOUDANT_DB_NAME,
            doc_id=doc_id
        ).get_result()
        # Delete using the revision
        cloudant_client.delete_document(
            db=CLOUDANT_DB_NAME,
            doc_id=doc_id,
            rev=doc['_rev']
        ).get_result()
        return True
    except Exception as e:
        print(f"Cloudant delete error: {e}")
        return False


# ── Gemini API helpers ─────────────────────────────────────────────────────────

def call_gemini_api(payload):
    """
    Calls the Gemini API using curl via subprocess.
    This avoids TLS compatibility issues with the system's older LibreSSL.
    """
    payload_json = json.dumps(payload)
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST", GEMINI_API_URL,
            "-H", "Content-Type: application/json",
            "-d", payload_json,
            "--max-time", "60"
        ],
        capture_output=True, text=True, timeout=90
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    return json.loads(result.stdout)


def generate_plan_with_gemini(location, days, budget):
    """
    Uses Google Gemini 2.5 API to generate a unique, detailed travel
    itinerary for any destination in the world.
    """
    if not location:
        return ["Please mention a destination in your query."]

    days = days if days else 3  # default to 3 days

    # Build a rich prompt for the Gemini model
    prompt = (
        f"You are an expert travel planner. Create a detailed and unique day-by-day travel itinerary "
        f"for {days} days in {location}."
    )
    if budget:
        prompt += f" The traveler has an approximate budget of {budget} INR."
    prompt += (
        "\n\nRules:\n"
        "- Start each day with 'Day X:' followed by a concise plan.\n"
        "- Include must-see attractions, local food recommendations, and hidden gems.\n"
        "- Keep each day's description to 1-2 sentences.\n"
        "- Do NOT use markdown formatting, bullet points, or asterisks.\n"
        "- Return ONLY the day-by-day plan, nothing else.\n"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 8192
        }
    }

    # Retry up to 3 times for transient errors (503, 429, etc.)
    last_error = None
    for attempt in range(1, 4):
        try:
            data = call_gemini_api(payload)

            # Check for API-level errors
            if "error" in data:
                err_msg = data["error"].get("message", "Unknown error")
                err_code = data["error"].get("code", 0)
                print(f"Gemini API error (attempt {attempt}): [{err_code}] {err_msg}")
                if err_code in (429, 503) and attempt < 3:
                    time.sleep(3 * attempt)
                    continue
                last_error = err_msg
                break

            # Extract the generated text from Gemini response
            generated_text = data["candidates"][0]["content"]["parts"][0]["text"]

            # Split into individual day lines and clean up
            lines = [line.strip() for line in generated_text.strip().split("\n") if line.strip()]
            plan = [line for line in lines if line.lower().startswith("day")]

            if not plan:
                plan = lines if lines else [f"Explore the best of {location}!"]

            return plan

        except Exception as e:
            print(f"Gemini API error (attempt {attempt}): {e}")
            last_error = str(e)
            if attempt < 3:
                time.sleep(3 * attempt)

    # Graceful fallback
    print(f"All Gemini API attempts failed. Last error: {last_error}")
    return [
        f"Day {i}: Explore famous attractions and local cuisine in {location.title()}"
        for i in range(1, days + 1)
    ]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    plan = []
    user_input = ""

    if request.method == "POST":
        text = request.form["user_input"]
        user_input = text

        response = nlu.analyze(
            text=text,
            features=Features(entities=EntitiesOptions())
        ).get_result()

        entities = response['entities']

        location = None
        days = None
        budget = None

        for entity in entities:
            if entity['type'] == 'Location':
                location = entity['text']
            elif entity['type'] == 'Number':
                value = int(entity['text'])
                if value <= 10:
                    days = value
                else:
                    budget = value

        # Use Gemini API for dynamic itinerary generation
        plan = generate_plan_with_gemini(location, days, budget)

        # Save to Cloudant
        save_to_cloudant(text, location, days, budget, plan)

    # Get search history
    history = get_history_from_cloudant()

    return render_template("index.html", plan=plan, history=history, user_input=user_input)


@app.route("/history", methods=["GET"])
def get_history():
    """API endpoint to fetch history as JSON."""
    history = get_history_from_cloudant()
    return jsonify(history)


@app.route("/history/<doc_id>", methods=["DELETE"])
def delete_history(doc_id):
    """API endpoint to delete a history entry."""
    success = delete_from_cloudant(doc_id)
    if success:
        return jsonify({"status": "deleted"})
    return jsonify({"status": "error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # Return JSON or text for non-HTTP errors
    import traceback
    return f"Internal Server Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))