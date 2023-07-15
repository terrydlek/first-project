from geopy.geocoders import Nominatim

def geocoding_reverse(lat_lng_str):
    geolocoder = Nominatim(user_agent='South Korea', timeout=None)
    address = geolocoder.reverse(lat_lng_str)
    return address

address = geocoding_reverse('36.77384887777778, 127.04070869722223')
print("주소: ", address)

address_list = address[0].split(",")
print("주소 리스트: ", address_list)

시도이름 = address_list[2].strip() + "_" + address_list[1].strip()
print("시도이름: ", 시도이름)
