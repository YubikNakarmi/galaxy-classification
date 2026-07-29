from PIL import Image
import io
import base64
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# img = root / "images" / "3.jpg"


# def image_to_base64(image_path, format="PNG"):
#     img = Image.open(image_path)
#     buffer = io.BytesIO()
#     img.save(buffer, format=format)
#     encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
#     return encoded

# encoded_str = image_to_base64(img)

# with open(root / "tests"/"image.txt", "w") as f:
#     f.write(encoded_str)


base_64_test = lambda: open(root / "tests"/"image.txt", "r").read()

images_list = [base_64_test() for _ in range(3)]  # Create a list of 3 base64-encoded images

files = [
        ("images", (f"img{i}.jpg", base64.b64decode(b), "image/jpeg"))
        for i, b in enumerate(images_list)
        ]