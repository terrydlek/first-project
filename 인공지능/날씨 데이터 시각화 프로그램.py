import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# 웹 페이지에 접속하여 HTML 데이터 가져오기
url = ''
response = requests.get(url)
html = response.text

# BeautifulSoup을 사용하여 HTML 파싱
soup = BeautifulSoup(html, 'html.parser')

# 필요한 데이터 추출
dates = []
temperatures = []
precipitations = []

# 예시: 날짜와 온도 데이터 추출
date_elements = soup.select('.date')
temperature_elements = soup.select('.temperature')

for date_element, temperature_element in zip(date_elements, temperature_elements):
    date = date_element.get_text()
    temperature = float(temperature_element.get_text().replace('°C', ''))

    dates.append(date)
    temperatures.append(temperature)

# 그래프 그리기
plt.plot(dates, temperatures)
plt.title('Weather Observations')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.show()
