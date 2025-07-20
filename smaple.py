import regex as re  # Use 'regex' module, NOT 're'

pattern = r"(\d{2}[A-Z]){e<=1}(\s+){e<=2}(\d{6}){e<=2}"  # Allow up to 2 errors
text = "4EU 110666"

match = re.search(pattern, text)

if match:
    print("Fuzzy match found:", match.group())
else:
    print("No match found")
