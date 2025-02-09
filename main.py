from base64 import b64encode
from base64 import b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
from Crypto.Random import get_random_bytes


class CipherUtil:
    def __init__(self, key: str):
        if isinstance(key, str):
            key = b64decode(key.encode())
            self.key = key

    def encrypt(self, text: str) -> str:
        iv = get_random_bytes(16)
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        cipher_text = aes.encrypt(pad(text.encode(), AES.block_size))
        return b64encode(iv + cipher_text).decode()

    def decrypt(self, text: str):
        encrypted_text = b64decode(text)
        iv = encrypted_text[:16]
        cipher_text = encrypted_text[16:]
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(aes.decrypt(cipher_text), AES.block_size).decode()


key_string = "rtvk7fsbT85y2ITmR6MJVxwyN9HDp1c53rSVAOh5VyI="
cipher_util = CipherUtil(key_string)

plaintext = "Sensitive data to encrypt"
encrypted_text = cipher_util.encrypt(plaintext)
print("Зашифрованный текст:", encrypted_text)

decrypted_text = cipher_util.decrypt(encrypted_text)
print("Расшифрованный текст:", decrypted_text)



