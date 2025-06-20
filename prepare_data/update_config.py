import sys
from lookup import get_class_label

label_name = sys.argv[1] if len(sys.argv) > 1 else None
label_code = get_class_label(label_name)

if label_code is None:
    raise ValueError(f"Label '{label_name}' not found in the class descriptions.")

with open("./prepare_data/config.py", "w") as f:
    f.write(f"CLASS_ID = '{label_code}'\n")

print(f"✅ Config updated: {label_name} ➝ {label_code}")
