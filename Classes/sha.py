import hashlib
import struct

# Your secret key or any random value
secret_key = b"MySecretKey"
def generat1e_henon_parameters(secret_key):
    hash_obj = hashlib.sha256(secret_key)
    hash_bytes = hash_obj.digest()[:16]

    # Split the hash into four 32-bit integers (x, y, a, b)
    x, y, a, b = struct.unpack('>IIII', hash_bytes)

    # Map x and y to the desired range (e.g., between -1 and 1)
    x = -1 + (x / 0xFFFFFFFF) * 2
    y = -1 + (y / 0xFFFFFFFF) * 2

    # Map a and b to the desired range (e.g., between 1 and 2)
    a = 1 + (a / 0xFFFFFFFF)
    b = 1 + (b / 0xFFFFFFFF)


