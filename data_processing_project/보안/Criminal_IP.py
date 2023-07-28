import requests

url = "https://api.criminalip.io/v1/ip/hosting"
api_key = ""

ip_address = input("Enter an IP address: ")
num_domains = int(input("Enter the number of domains to display: "))

headers = {
    "x-api-key": api_key
}
