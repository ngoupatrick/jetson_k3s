# python3 /home/patrick/Bureau/vpn/jetson_k3s/models/expose_models/python_test/test1.py
import requests
resp = requests.post("http://picomaster:31080/predict", files={"image": open("/home/patrick/Bureau/test/water-citron.jpg", "rb")})
print(resp.json())
