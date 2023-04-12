import requests
import json
import sys
import os
import hashlib
import hmac
import base64
import time
timestamp = int(time.time() * 1000)
timestamp = str(timestamp)

access_key = "Du14E52SZ3CxQVdLs4ng"
secret_key = "u5JcMiDHFBwn5A8inIdtdpIJcYph194d6b8uSLmm"
url = "https://sens.apigw.ntruss.com"
uri = "/sms/v2/services/ncp:sms:kr:306057061747:capstone/messages"

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
    "from":"01039061652",
    "content": "hi",
    "subject":"SENS",
    "messages":[
        {
            "to": "01089643268",
        }
    ]
}


res = requests.post(url+uri, headers=header, data=json.dumps(data))
print(res.text)
