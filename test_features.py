from feature_extraction import extract_features


test_url = "https://www.example.com/login/account"

features = extract_features(test_url)

print("URL:")
print(test_url)

print("\nExtracted Features:")

for feature, value in features.items():
    print(f"{feature}: {value}")