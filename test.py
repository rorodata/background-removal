"""Script to call the background-removal API with an image.

USAGE: python test.py static/bird.jpg
"""
import firefly
import sys
import shutil

background_removal = firefly.Client("https://background-removal.rorocloud.io/")

def main():
    image_path = sys.argv[1]
    format = image_path.split(".")[-1]

    print("calling the background removal API...")
    new_image = background_removal.predict(image_file=open(image_path, 'rb'), format=format)

    ## To open Image using PIL
    # from PIL import Image
    # img = Image.open(new_image)

    with open("output.jpg", "wb") as f:
        shutil.copyfileobj(new_image, f)
    print("saved the output image in output.jpg")

if __name__ == '__main__':
    main()
