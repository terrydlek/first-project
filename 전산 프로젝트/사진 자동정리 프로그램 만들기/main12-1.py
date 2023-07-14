from PIL import Image
from PIL.ExifTags import TAGS
from glob import glob
import os # 파일 관리 기능
os.chdir(os.path.dirname(os.path.abspath(__file__)))

images = (glob(r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\사진 자동정리 프로그램 만들기\사진\*.jpg'))
images.extend(glob(r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\사진 자동정리 프로그램 만들기\사진\*.png'))

print("사진들:", images)

image = Image.open(images[0])
info = image._getexif()
image.close()

taglabel = {}
for tag, value in info.items():
    decoded = TAGS.get(tag, tag)
    taglabel[decoded] = value

print("사진정보: ", taglabel)
print("사진촬영날짜: ", taglabel['DateTime'])
print("사진촬영장소: ", taglabel['GPSInfo'])
