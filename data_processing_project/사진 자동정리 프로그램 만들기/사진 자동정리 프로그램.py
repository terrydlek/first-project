from PIL import Image
from PIL.ExifTags import TAGS
from geopy.geocoders import Nominatim
from glob import glob

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

사진들 = (glob(r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\사진 자동정리 프로그램 만들기\사진\*.jpg'))
사진들.extend(glob(r'C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\사진 자동정리 프로그램 만들기\사진\*.png'))

image = Image.open(사진들[0])
info = image._getexif()
image.close()

# 위도, 경도 넣으면 주소 나옴
def geocoding_reverse(lat_lng_str):
    geolocoder = Nominatim(user_agent='South Korea', timeout=None)
    address = geolocoder.reverse(lat_lng_str)
    return address


for 사진 in 사진들:
    image = Image.open(사진)
    info = image._getexif();
    image.close()

    taglabel = {}
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        taglabel[decoded] = value

    위도 = (((taglabel['GPSInfo'][2][2] / 60.0) + taglabel['GPSInfo'][2][1]) / 60.0) + taglabel['GPSInfo'][2][0]
    경도 = (((taglabel['GPSInfo'][4][2] / 60.0) + taglabel['GPSInfo'][4][1]) / 60.0) + taglabel['GPSInfo'][4][0]
    address = geocoding_reverse(str(위도) + "," + str(경도))
    address_list = address[0].split(',')
    시도이름 = ""
    if len(address_list) == 6:
        시도이름 = address_list[3].strip() + "_" + address_list[2].strip()
    elif len(address_list) == 5:
        시도이름 = address_list[2].strip() + "_" + address_list[1].strip()
    print("시도이름: ", 시도이름)

    사진촬영시간 = taglabel['DateTime']
    사진이름_변경 = "사진\\" + 시도이름 + "_" + 사진촬영시간.replace(":", "-") + "." + 사진.split(".")[1]

    print(사진이름_변경)

    os.rename(사진, 사진이름_변경)