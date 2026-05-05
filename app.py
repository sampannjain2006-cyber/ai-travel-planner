from flask import Flask, render_template, request
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions

app = Flask(__name__)

# IBM NLU setup
import os

apikey = os.environ.get("NLU_API_KEY") or "QQk4WAzoL6tAvGE5SKvf9bHW_umYN4wkWcxNQmmcBE9V"
url = os.environ.get("NLU_URL") or "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/d75430ab-dcea-4908-af01-9f20c67ffa80"
authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(version='2021-08-01', authenticator=authenticator)
nlu.set_service_url(url)


def generate_plan(location, days, budget):
    plan = []

    location = location.lower() if location else ""
    days = days if days else 1   # ✅ prevents crash

    if location == "goa":
        if days >= 1:
            plan.append("Day 1: Baga Beach, Calangute Beach, Night market")
        if days >= 2:
            plan.append("Day 2: Fort Aguada, Anjuna Beach, Water sports")
        if days >= 3:
            plan.append("Day 3: Dudhsagar Falls, Spice plantation")

    elif location == "manali":
        if days >= 1:
            plan.append("Day 1: Mall Road, Local cafes")
        if days >= 2:
            plan.append("Day 2: Solang Valley, Adventure sports")
        if days >= 3:
            plan.append("Day 3: Hidimba Temple, Old Manali")

    elif location == "mumbai":
        if days >= 1:
            plan.append("Day 1: Gateway of India, Marine Drive, Colaba")
        if days >= 2:
            plan.append("Day 2: Elephanta Caves, Juhu Beach")
        if days >= 3:
            plan.append("Day 3: Bandra Bandstand, Shopping at Linking Road")

    else:
        # ✅ Smart fallback (works for ANY city)
        for i in range(1, days + 1):
            plan.append(f"Day {i}: Explore famous places and local food in {location.title()}")

    return plan


@app.route("/", methods=["GET", "POST"])
def home():
    plan = []

    if request.method == "POST":
        text = request.form["user_input"]

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

        plan = generate_plan(location, days, budget)

    return render_template("index.html", plan=plan)


if __name__ == "__main__":
    import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))