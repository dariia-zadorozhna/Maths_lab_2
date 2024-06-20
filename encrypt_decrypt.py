import numpy as np


def encrypt_message(message, key_matrix):

    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonal_matrix = np.diag(eigenvalues)

    # Diagonalized key matrix
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, diagonal_matrix), np.linalg.inv(eigenvectors))

    # Inverse of the diagonalized key matrix
    inv_diag_key_matrix = np.linalg.inv(diagonalized_key_matrix)

    # Decrypt the encrypted vector
    decrypted_vector = np.dot(inv_diag_key_matrix, encrypted_vector)

    # Convert decrypted vector to string
    decrypted_message = ''.join([chr(int(round(num))) for num in decrypted_vector.real])

    return decrypted_message


if __name__ == "__main__":
    message = "Hello, World!"
    key_matrix = np.random.randint(0, 256, (len(message), len(message)))
    encrypted_vector = encrypt_message(message, key_matrix)
    print("Original message:", message)
    print("Encrypted message:", encrypted_vector)
    decrypt_message = decrypt_message(encrypted_vector, key_matrix)
    print("Decrypted message:", decrypt_message)