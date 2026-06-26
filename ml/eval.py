from pathlib import Path

root = Path(__file__).parent.parent

p = root / Path("models/temp/resume_best.pt")

print("Exists:", p.exists())
print("Is file:", p.is_file())
print("Is dir:", p.is_dir())