from Crypto.Random import get_random_bytes
import base64

key = get_random_bytes(32)

print("Сгенерированный ключ (base64):")
print(base64.b64encode(key).decode())

