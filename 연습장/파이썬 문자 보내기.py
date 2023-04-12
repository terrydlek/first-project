import requests
import json
import hashlib
import hmac
import base64
import time
timestamp = int(time.time() * 1000)
timestamp = str(timestamp)

access_key =
secret_key =
url =
uri =


def make_signature(secret_key, access_key, timestamp, uri):
    secret_key = bytes(secret_key, 'UTF-8')
    method = "POST"
    message = method + " "  + uri + "\n" + timestamp + "\n" + access_key
    message = bytes(message, 'UTF-8')
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey


header = {
    "Content-Type": "application/json; charset=utf-8",
    "x-ncp-apigw-timestamp": timestamp,
    "x-ncp-iam-access-key": access_key,
    "x-ncp-apigw-signature-v2": make_signature(secret_key, access_key, timestamp, uri)
}


data = {
    "type":"SMS",
    "from":"",
    "content": "hi",
    "subject":"SENS",
    "messages":[
        {
            "to": "",
        }
    ]
}


res = requests.post(url+uri, headers=header, data=json.dumps(data))
print(res.text)
