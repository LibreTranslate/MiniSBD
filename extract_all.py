import subprocess
import glob
import os
from minisbd.models import list_models

processed = set([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join("onnx", "*.onnx"))])
all_models = set(list_models())
remaining = all_models - processed
fail_langs = []

for lang_code in remaining:
    result = subprocess.run(["python", "extract.py", "--lang-code", lang_code])
    if result.returncode == 1:
        fail_langs.append(lang_code)

if fail_langs:
    print("Cannot extract:", fail_langs)