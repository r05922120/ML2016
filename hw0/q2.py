import sys
from PIL import Image
im = Image.open(sys.argv[1])
im2 = im.rotate(180)
im2.save("ans2.png")
