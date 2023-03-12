from PIL import Image

image_1 = 'img/1.jpg'
image_1 = Image.open(image_1)
image_1.rotate(270).show()
