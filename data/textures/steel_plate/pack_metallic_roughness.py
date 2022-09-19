from PIL import Image, ImageDraw

roughness_im = Image.open("roughness.png")
metalness_im = Image.open("metalness.png")
ambient_occlusion_im = Image.open("ao.png")

#These are monocolor maps with the same data in each channel
r_channel = roughness_im.getdata(0)
m_channel = metalness_im.getdata(0)
ao_channel = ambient_occlusion_im.getdata(0)

out_image = Image.new("RGBA", roughness_im.size, (128, 128, 128, 255))
drawer = ImageDraw.Draw(out_image)

print(len(r_channel))
print(len(m_channel))

points_to_draw = []
pixel_count = len(r_channel)
for i in range(0, pixel_count):
	x = i % roughness_im.size[0]
	y = i / roughness_im.size[0]
	drawer.point((x, y), fill=(ao_channel[i], r_channel[i], m_channel[i], 255))

out_image.save("ao_roughness_metallic.png", "PNG")
print("Done!")