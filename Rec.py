
from numpy import size
import Crypto
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

#utiliza un generador pseudoaleatorio criptográficamente seguro (CSPRNG) .









def getData(n):
    random_generator = Crypto.Random.new().read
    private_key = RSA.generate(1024, random_generator)
    public_key = private_key.publickey()


    name = ["Aranza", "Connie", "Gaby"]
    email = ["Mary001@gmail.com", "Connie001@gmail.com", "Gaby001@gmail.com"]
    phone = [44421474983, 44421474985, 4421474986]

    message = ""
    for x in range(size(name)):
        if n == name[x]:
            message = name[x]+"," +email[x]+"," + str(phone[x])
    message = message.encode()

    cipher = PKCS1_OAEP.new(public_key)
    encrypted_message = cipher.encrypt(message)



    return encrypted_message

a="Aranza"
getData(a)






