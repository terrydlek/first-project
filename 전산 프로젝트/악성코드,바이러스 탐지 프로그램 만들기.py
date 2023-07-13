import requests

api_key = '5D7F30B9145B41DD888A72A24C9C9A7AA671995FE0EF4F1189AAFA20179EF4E4'


params = {'api_key': api_key, 'hash': '111BFB224BAADB8016'}
response = requests.get('https://public.api.malwares.com/api/v22/file/upload', params=params)
json_response = response.json()
print(json_response)
