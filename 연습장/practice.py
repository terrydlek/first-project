import hmac, hashlib, base64
import time, requests, json


def make_signature(secret_key, access_key, timestamp, uri):
    secret_key = bytes(secret_key, 'UTF-8')
    method = "POST"
    message = method + " "  + uri + "\\n" + timestamp + "\\n" + access_key
    message = bytes(message, 'UTF-8')
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey

access_key = "Du14E52SZ3CxQVdLs4ng"
secret_key = "u5JcMiDHFBwn5A8inIdtdpIJcYph194d6b8uSLmm"
service_key = "ncp:sms:kr:306057061747:capstone"
#service_key = "https://ncp:sms:kr:306057061747:capstone"

# <https://api.ncloud-docs.com/docs/ko/ai-application-service-sens-smsv2>
url = "https://sens.apigw.ntruss.com"
uri = f"/sms/v2/services/{service_key}/messages"

timestamp = int(time.time() * 1000)
timestamp = str(timestamp)

# 받는 상대방
number = "01089643268"

# message 내용
contents = "test sms"

header = {
    'requestId': "terryf618@naver.com",
    "Content-Type": "application/json; charset=utf-8",
    "x-ncp-apigw-timestamp": timestamp,
    "x-ncp-iam-access-key": access_key,
    "x-ncp-apigw-signature-v2": make_signature(secret_key, access_key, timestamp, uri)
}

# from : SMS 인증한 사용자만 가능
data = {
    'requestId': "terryf618@naver.com",
    "type":"SMS",
    "from":"01039061652",
    "content":contents,
    "subject":"SENS",
    "messages":[
        {
            "to":number,
        }
    ]
}

res = requests.post(url+uri,headers=header,data=json.dumps(data))
datas = json.loads(res.text)
reid = datas['requestId']

print("메시지 전송 상태")
print(res.text+"\\n")