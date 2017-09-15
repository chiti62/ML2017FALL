from PIL import Image
import sys
import math

im = Image.open(sys.argv[1])

#print (im.format, im.size, im.mode)

width, height = im.size
ans = Image.new("RGB",(width,height))

for w in range(width):
	for h in range(height):
		r, g, b = im.getpixel((w, h))
		r = math.floor(r/2)
		g = math.floor(g/2)
		b = math.floor(b/2)
		ans.putpixel([w,h],(r,g,b))
ans.save("Q2.jpg")
