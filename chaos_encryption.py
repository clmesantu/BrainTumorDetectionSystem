# chaos_encryption.py
import numpy as np
import secrets
import hashlib

def generate_key():
    key = secrets.token_bytes(16)  # 128-bit key
    return key.hex()

def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def generate_chaos_sequence(length, seed):
    sequence = []
    x = seed
    for _ in range(length):
        x = logistic_map(x)
        sequence.append(int(x * 256) % 256)
    return np.array(sequence, dtype=np.uint8)

def encrypt_image(image, hex_key):
    key_bytes = bytes.fromhex(hex_key)
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest(), 'big') % 1000 / 1000
    flat_image = image.flatten()
    chaos_seq = generate_chaos_sequence(flat_image.size, seed)
    encrypted_flat = np.bitwise_xor(flat_image, chaos_seq)
    encrypted_image = encrypted_flat.reshape(image.shape)
    return encrypted_image

def decrypt_image(encrypted_image, hex_key):
    return encrypt_image(encrypted_image, hex_key)  # XOR again to decrypt