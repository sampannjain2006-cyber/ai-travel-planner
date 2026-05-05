from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions

apikey = "QQk4WAzoL6tAvGE5SKvf9bHW_umYN4wkWcxNQmmcBE9V"
url = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/d75430ab-dcea-4908-af01-9f20c67ffa80"

authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(version='2021-08-01', authenticator=authenticator)
nlu.set_service_url(url)

text = "Plan a 3 day trip to Goa under 8000"

response = nlu.analyze(
    text=text,
    features=Features(entities=EntitiesOptions())
).get_result()

print(response)
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

print("\n--- Extracted Data ---")
print("Location:", location)
print("Days:", days)
print("Budget:", budget)
def generate_plan(location, days, budget):
    plan = []

    if location and location.lower() == "goa":
        if days >= 1:
            plan.append("Day 1: Baga Beach, Calangute Beach, Night market")
        if days >= 2:
            plan.append("Day 2: Fort Aguada, Anjuna Beach, Water sports")
        if days >= 3:
            plan.append("Day 3: Dudhsagar Falls, Spice plantation")

    elif location and location.lower() == "manali":
        if days >= 1:
            plan.append("Day 1: Mall Road, Local cafes")
        if days >= 2:
            plan.append("Day 2: Solang Valley, Adventure sports")
        if days >= 3:
            plan.append("Day 3: Hidimba Temple, Old Manali")

    else:
        plan.append("Explore local attractions and enjoy local food")

    return plan
print("\n--- Travel Plan ---")

trip_plan = generate_plan(location, days, budget)

for day in trip_plan:
    print(day)