import json
from numpy import size
#get_ipython().system('pip install opencv-contrib-python')
#pip install matplotlib

import Crypto
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

def fun(nombre):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    path = 'C:/Users/edsel/Documents/Hackathon 2020/Reconocimiento en R/Reconocimiento_facial_en_Python-master/Conocidos/'
    DIR_CONOCIDOS = path + nombre
    print(DIR_CONOCIDOS)
    #DIR_CONOCIDOS = 'C:/Users/edsel/Documents/Hackathon 2020/Reconocimiento en R/Reconocimiento_facial_en_Python-master/Conocidos' #Cambia a la ruta donde hayas almacenado Data
    DIR_DESCONOCIDOS = 'C:/Users/edsel/Documents/Hackathon 2020/Reconocimiento en R/Reconocimiento_facial_en_Python-master/Desconocidos'
    DIR_RESULTADOS = 'C:/Users/edsel/Documents/Hackathon 2020/Reconocimiento en R/Reconocimiento_facial_en_Python-master/Resultados'
    DIR_ELLOS = "C:/Users/edsel/Documents/Hackathon 2020/Reconocimiento en R/Reconocimiento_facial_en_Python-master/static/uploads"

    # In[3]:


    # Leer mobilenet_graph.pb
    with tf.io.gfile.GFile('mobilenet_graph.pb','rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as mobilenet:
        tf.import_graph_def(graph_def,name='')


    # In[4]:


    print(mobilenet)


    # In[5]:


    # Cargar imagen
    def load_image(DIR, NAME):
        return cv2.cvtColor(cv2.imread(f'{DIR}/{name}'), cv2.COLOR_BGR2RGB)


    # In[6]:


    def detect_faces(image, score_threshold=0.7):
        global boxes, scores
        (imh, imw) = image.shape[:-1]
        img = np.expand_dims(image,axis=0)

        # Inicializar mobilenet
        sess = tf.compat.v1.Session(graph=mobilenet)
        image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
        boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
        scores = mobilenet.get_tensor_by_name('detection_scores:0')

        # Predicción (detección)
        (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor:img})

        # Reajustar tamaños boxes, scores
        boxes = np.squeeze(boxes,axis=0)
        scores = np.squeeze(scores,axis=0)

        # Depurar bounding boxes
        idx = np.where(scores>=score_threshold)[0]

        # Crear bounding boxes
        bboxes = []
        for index in idx:
            ymin, xmin, ymax, xmax = boxes[index,:]
            (left, right, top, bottom) = (xmin*imw, xmax*imw, ymin*imh, ymax*imh)
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            bboxes.append([left,right,top,bottom])

        return bboxes


    # In[7]:


    # Dibujar bounding boxes
    def draw_box(image,box,color,line_width=6):
        if box==[]:
            return image
        else:
            cv2.rectangle(image,(box[0],box[2]),(box[1],box[3]),color,line_width)
        return image


    # In[40]:


    name = 'Ga_N02.jpeg'
    image = load_image(DIR_DESCONOCIDOS,name)
    bboxes = detect_faces(image)
    for box in bboxes:
        detected_faces = draw_box(image,box,(0,255,0))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(detected_faces)


    # In[41]:


    # Extraer rostros
    def extract_faces(image,bboxes,new_size=(160,160)):
        cropped_faces = []
        for box in bboxes:
            left, right, top, bottom = box
            face = image[top:bottom,left:right]
            cropped_faces.append(cv2.resize(face,dsize=new_size))
        return cropped_faces


    # In[42]:


    faces = extract_faces(image,bboxes)
    plt.imshow(faces[0])


    # In[13]:


    # FaceNet
    facenet = load_model('facenet_keras.h5')
    print(facenet.input_shape)
    print(facenet.output_shape)


    # In[14]:


    def compute_embedding(model,face):
        face = face.astype('float32')

        mean, std = face.mean(), face.std()
        face = (face-mean) / std

        face = np.expand_dims(face,axis=0)

        embedding = model.predict(face)
        return embedding


    # In[43]:


    embedding = compute_embedding(facenet,faces[0])
    print(embedding)


    # In[44]:


    # Embeddings referencia
    known_embeddings = []

    print('Procesando rostros conocidos...')
    for name in os.listdir(DIR_CONOCIDOS):
        if name.endswith('.jpeg'):
            print(f'   {name}')
            image = load_image(DIR_CONOCIDOS,name)
            bboxes = detect_faces(image)
            face = extract_faces(image,bboxes)
            known_embeddings.append(compute_embedding(facenet,face[0]))



    # In[45]:


    print(known_embeddings)


    # In[46]:


    def compare_faces(embs_ref, emb_desc, umbral=11):
        distancias = []
        for emb_ref in embs_ref:
            distancias.append(np.linalg.norm(emb_ref-emb_desc))
        distancias = np.array(distancias)
        return distancias, list(distancias<=umbral)


    # In[49]:


    var = "0"
    # Reconocimiento (????) de rostros desconocidos
    print('Procesando imágenes desconocidas...')
    for name in os.listdir(DIR_ELLOS):
        if name.endswith('.jpg'):
            print(f'   {name}')
            image = load_image(DIR_ELLOS,name)
            bboxes = detect_faces(image)
            faces = extract_faces(image,bboxes)

            # Por cada rostro calcular embedding
            img_with_boxes = image.copy()
            for face, box in zip(faces,bboxes):
                emb = compute_embedding(facenet,face)

                _, reconocimiento = compare_faces(known_embeddings,emb)

                if any(reconocimiento):
                    print('     match!')
                    img_with_boxes = draw_box(img_with_boxes,box,(0,255,0))
                    return "1"
                else:
                    var = "0"
                    img_with_boxes = draw_box(img_with_boxes,box,(255,0,0))


            cv2.imwrite(f'{DIR_RESULTADOS}/{name}',cv2.cvtColor(img_with_boxes,cv2.COLOR_RGB2BGR))
    print('¡Fin!')
    return var


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
            message = name[x] + "," + email[x] + "," + str(phone[x])
    message = message.encode()

    cipher = PKCS1_OAEP.new(public_key)
    encrypted_message = cipher.encrypt(message)

    return encrypted_message

# In[ ]:




