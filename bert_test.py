from transformers import pipeline

# Load model
classifier = pipeline("zero-shot-classification")

# FIR description
text = "Two people attacked me and stole my phone"

# IPC-based labels (IMPORTANT)
labels = [
    "IPC 379 - Theft",
    "IPC 392 - Robbery",
    "IPC 323 - Voluntarily causing hurt",
    "IPC 420 - Cheating",
    "IPC 506 - Criminal intimidation"
]

# Run model
result = classifier(text, labels)

print("\nInput:", text)
print("\nSuggested IPC Sections:")

for label, score in zip(result['labels'], result['scores']):
    print(f"{label} → {round(score*100, 2)}%")