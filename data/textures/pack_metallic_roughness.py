from PIL import Image

with Image.open("roughness.png") as im:
	r_channel = im.getdata(0)