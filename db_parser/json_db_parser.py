import os
import json
import numpy as np
from collections import defaultdict

class Submission:
    def __init__(self, uuid):
        self.uuid = uuid
        self.quality = defaultdict(dict)
        self.temporal = defaultdict(dict)
    
    def __repr__(self):
        return f"Submission(uuid={self.uuid}, quality={dict(self.quality)}, temporal={dict(self.temporal)})"

def parse_json(data):
    submissions = defaultdict(lambda: Submission(None))
    for key, entry in data["perceptual_audio_quality_test"].items():
        uuid = entry["questionaire"]["uuid"]
        category = entry["id"].split('_')[-1]  # Extracts the category (e.g., "quality" or "temporal")
        event_type = "_".join(entry["id"].split('_')[:-1])  # Extracts the event type (e.g., "dog_barking")
        
        if submissions[uuid].uuid is None:
            submissions[uuid].uuid = uuid
        
        section = submissions[uuid].quality if category == "quality" else submissions[uuid].temporal
        section[event_type] = entry["responses"]
    
    return list(submissions.values())

def compute_means(submissions):
    # Dictionary to hold mean values
    means = {
        "quality": defaultdict(lambda: defaultdict(list)),  # category -> model -> values
        "temporal": defaultdict(lambda: defaultdict(list))   # category -> model -> values
    }
    
    # Gather all responses into means dictionary
    for submission in submissions:
        # For quality responses
        for category, responses in submission.quality.items():
            for entry in responses:
                model = entry['stimulus']
                score = entry['score']
                means["quality"][category][model].append(score)

        # For temporal responses
        for category, responses in submission.temporal.items():
            for entry in responses:
                model = entry['stimulus']
                score = entry['score']
                means["temporal"][category][model].append(score)

    # Compute mean values for each category
    mean_values = {
        "quality": defaultdict(dict),
        "temporal": defaultdict(dict)
    }
    
    for section in means:
        for category, models in means[section].items():
            for model, values in models.items():
                if values:  # Avoid division by zero
                    mean_values[section][category][model] = np.mean(values)
                else:
                    mean_values[section][category][model] = None  # No values

    # Compute overall means across all categories for each model and section
    overall_means = {
        "quality": defaultdict(dict),
        "temporal": defaultdict(dict)
    }

    for section in mean_values:
        for model in set(model for category in mean_values[section] for model in mean_values[section][category]):
            all_scores = []
            for category in mean_values[section]:
                if model in mean_values[section][category]:
                    score = mean_values[section][category][model]
                    if score is not None:
                        all_scores.append(score)
            # Calculate the mean across all categories for this model
            if all_scores:
                overall_means[section][model] = np.mean(all_scores)
            else:
                overall_means[section][model] = None  # No data

    return mean_values, overall_means




file_path = os.path.realpath("new_file.json")
# Load the JSON data
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Parse the JSON data into the desired structure
submissions = parse_json(json_data)

# Print the parsed submissions at all levels
# for submission in submissions:
#     print(submission.uuid)
#     print(submission.quality)
#     print(submission.temporal)
#     for section, stimulus in submission.quality.items():
#         print(f"\nQuality for {section}: {stimulus}")
#         for model in stimulus:
#             print(f"{model}")
#             for key, value in model.items():
#                 print(f"{key}: {value}")
#     print("")



# Compute the mean values
mean_values, overall_means = compute_means(submissions)

# Print the computed mean values
for section in mean_values:
    print(f"Mean Values for {section.capitalize()}:")
    for category, models in mean_values[section].items():
        print(f"  Category: {category}")
        for model, mean in models.items():
            print(f"    Model: {model}, Mean: {mean:.2f}" if mean is not None else f"    Model: {model}, Mean: No data")
    print("")

# Print overall means
for section in overall_means:
    print(f"Overall Mean Values for {section.capitalize()}:")
    for model, mean in overall_means[section].items():
        print(f"  Model: {model}, Overall Mean: {mean:.2f}" if mean is not None else f"  Model: {model}, Overall Mean: No data")
    print("")