from keras.models import load_model
from time import sleep
from keras.utils.image_utils import img_to_array
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
import cv2
import numpy as np
import os
import json

# for sending mail
import ssl
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime. base import MIMEBase 
from email import encoders

#my information please change with your mail and password
user = 'abcde@gmail.com'
password = 'your_password' 

# use input()
alici = input('Mail adresinizi giriniz: ')
baslik = 'Hacettepe Yapay Zeka Topluluğu'
#mesaj = 'Deneme Mesaji'
#context = ssl.create_default_context()

#for calculating probabilities
goodGuyCounter = 0
badGuyCounter = 0
labelDic = {'Heart': 0, 'Oblong' :0, 'Oval' : 0, 'Round' : 0, 'Square' : 0}

# opencv shortcuts, don't forget to change path
face_classifier = cv2.CascadeClassifier('C:\\Users\\baris\\Desktop\\ai projects\\emotionPrediction\\haarcascade_frontalface_default.xml')

# importing models, I will add links to download these models
from keras.models import load_model
model_path = "C:\\Users\\baris\\Downloads\\myModel2.h5"
classifier = load_model(model_path)
classifier2 = load_model("C:\\Users\\baris\\Downloads\\faceShapeModel.h5")

# labels that is created based on models outputs
class_labels=['Good Guy','Bad Guy']
class_labels2 = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
cap=cv2.VideoCapture(0)


# starting webcam 
while True:
    ret, frame = cap.read()
    labels = []
    labels2 = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # if face detected do this
    if len(faces) > 0:
        # Calculate the center of the frame
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # Find the closest face to the center
        closest_face = None
        min_distance = float('inf')
        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            distance = ((frame_center_x - face_center_x) ** 2 + (frame_center_y - face_center_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_face = (x, y, w, h)
        # we will use these coordinates for models
        x, y, w, h = closest_face
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        
        # Resize the input to match MobileNetV2's requirements
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi_resized = cv2.resize(roi_rgb, (224, 224))
        # Preprocess the input
        roi_preprocessed = preprocess_input(roi_resized)
        roi_preprocessed = cv2.resize(roi_rgb, (200, 200))


        # each code for second model
        roi_gray2 = gray[y:y + h, x:x + w]
        roi_rgb2 = cv2.cvtColor(roi_gray2, cv2.COLOR_GRAY2RGB)
        roi_resized2 = cv2.resize(roi_rgb2, (224, 224))
        roi_preprocessed2 = preprocess_input(roi_resized2)
        roi_preprocessed2 = cv2.resize(roi_rgb2, (150, 150))

        if np.sum([roi_preprocessed]) != 0 and np.sum([roi_preprocessed2]) != 0:
            roi_preprocessed = np.expand_dims(roi_preprocessed, axis=0)
            roi_preprocessed2 = np.expand_dims(roi_preprocessed2, axis = 0)
            preds = classifier.predict(roi_preprocessed)[0]
            preds2 = classifier2.predict(roi_preprocessed2)[0]

            label = class_labels[preds.argmax()]
            label2 = class_labels2[preds2.argmax()]

            label_position = (x, y)
            description = f'Emotion: {label}, Face Shape: {label2}'
            cv2.putText(frame, description, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if label == 'Good Guy':
                goodGuyCounter += 1
            else:
                badGuyCounter += 1
            
            labelDic[label2] += 1            

        else:
            cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if goodGuyCounter > badGuyCounter:
    firstVal = 'Good Guy'
    res = goodGuyCounter / (badGuyCounter + goodGuyCounter) * 100
    print(f'probability of being Good Guy is {res}')
else:
    firstVal = 'Bad Guy'
    res = badGuyCounter / (badGuyCounter + goodGuyCounter) * 100
    print(f'probability of being Bad Guy is {res}')


res2 = max(labelDic, key=lambda k: labelDic[k])
res3 = max(labelDic.values()) / (sum(labelDic.values())) * 100
print(f'being {res2} probability is {res3}')
print(labelDic)



if firstVal == 'Good Guy' and res2 == 'Heart':
    mesaj = f"""\
Probability Bf being Good Guy is {res}
Being {res2} face shape probability is {res3}

İyi niyetli ve kalp yüzlü bir kişi:
Kalp şeklinde bir yüze sahip bir kişi, "iyi niyetli" olarak tanımlandığında sıcak, içten ve şefkatli olarak görülebilir. İşte kalp şeklinde yüzü olan bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

Empatik: Kalp şeklinde bir yüz, başkalarının duygularına ve ihtiyaçlarına hassas ve duyarlı bir doğayı yansıtabilir. Diğer insanların duygularına karşı duyarlı biri olarak algılanabilirler.

Dostça: Kalp şeklindeki yüzler, yaklaşılabilir ve arkadaş canlısı olarak görünebilir, bu da onları başkalarıyla kolayca iletişim kurulabilen kişiler yapar.

Romantik: Kalp şeklindeki yüz, romantizm ve sevgi dolu bir tavır ile ilişkilendirilebilir, duygusal bağlantıları ve ilişkileri önemseyen bir kişiyi ima edebilir.

Duyarlı: Kalp şeklinde yüzü olan kişiler, çevrelerindeki insanların duygularına karşı hassas oldukları şeklinde algılanabilirler ve duygusal zeka seviyeleri yüksek olabilir.

Hayırsever: Kalp şeklinde yüze sahip bir kişi, başkalarına yardım etmeyi ve topluma geri vermeği seven biri olarak görülebilir.

Olumlu: Yüz yapısı genel olarak olumlu ve iyimser bir tavır katkıda bulunabilir.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
    """
    context = ssl.create_default_context()
elif firstVal == 'Good Guy' and res2 == 'Oblong':
    mesaj = f"""\
Probability of being Good Guy is {res}
Being {res2} face shape probability is {res3}

Dikdörtgen yüz şekli olan bir kişi, "iyi niyetli" olarak tanımlandığında zeki, yaklaşılabilir ve güvenilir olarak algılanabilir. İşte bir iyi insan imajına sahip biri ve Dikdörtgen yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

Zeka: Dikdörtgen yüz şekli genellikle zeka izlenimi yaratır ve bireyin düşünceli ve analitik olduğunu düşündürebilir.

Yaklaşılabilir: Dikdörtgen yüze sahip insanlar, yaklaşılabilir ve dostça görünebilir, bu da diğer insanların onlarla kolayca iletişim kurmasını sağlar.

Güvenilirlik: Dikdörtgen yüz, güvenilirlik ve güvenilirlik hissi ile ilişkilendirilebilir, bu da bir "iyi niyetli" olmakla uyumlu bir özelliktir.

Sakin ve Toparlanmış: Dikdörtgen yüzler sakin ve toparlanmış bir görünüm sergileyebilir, bu da kişinin zorlu durumları zarafetle ele alabileceğini gösterir.

Çalışkan: Bu yüz şekli, çalışkan ve işine sadık bir kişilikle ilişkilendirilebilir, sorumluluklarına bağlı biri olarak tanımlanabilirler.

Sorumlu: Dikdörtgen yüze sahip bireyler, sorumlu ve düzenli olarak algılanabilirler, bu da "iyi niyetli" imajlarına katkıda bulunur.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Good Guy' and res2 == 'Oval':
    mesaj = f"""\
Probability of being Good Guy is {res}
Being {res2} face shape probability is {res3}

Oval yüz şekli olan bir kişi, "iyi niyetli" olarak tanımlandığında dengeli, dostça ve yaklaşılabilir olarak algılanabilir. Bir iyi niyetli imajına ve oval yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

Dostça: Oval yüzler genellikle yaklaşılabilir ve dostça olarak görülür, bu da diğer insanların onlarla kolayca bağlantı kurabilmesini sağlar.

Dengeli: Oval yüz şekli dengeli ve uyumlu olarak kabul edilir, bu da dengeli ve sakin bir kişiliği yansıtabilir.

Sosyal: Oval yüzlü bireyler sosyal ve dışa dönük olarak algılanabilirler, başkalarıyla etkileşimden keyif alan biri olarak tanımlanabilirler.

Empatik: Oval yüzlü insanlar empatik ve anlayışlı olarak görülebilirler, çevrelerindekilerin duygularına önem veren bir tutum sergileyebilirler.

Rahat: Oval bir yüz, rahatlık ve uyum sağlama hissi ile iletebilir, bu da rahat ve uyumlu bir kişi olduğunu düşündürebilir.

Diplomatik: Oval yüz şekline sahip bireyler iletişim ve çatışma çözme konusunda başarılı olabilirler, bu da bir "iyi niyetli" imajıyla uyumlu bir şekilde uyumu teşvik eden bir kişiyi yansıtabilir.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Good Guy' and res2 == 'Round':
    mesaj = f"""\
Probability of being Good Guy is {res}
Being {res2} face shape probability is {res3}

Yuvarlak yüz şekli olan bir kişi, "iyi insan" olarak tanımlandığında dostça, yaklaşılabilir ve neşeli olarak algılanabilir. İşte bir iyi insan imajına sahip biri ve yuvarlak yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

Dostça: Yuvarlak yüzler genellikle sıcak ve dostça bir tavır ile iletilir, bu da diğer insanların onların etrafında rahat hissetmelerini kolaylaştırır.

Yaklaşılabilir: Yuvarlak yüzlü insanlar yaklaşılabilir ve açık olarak görünebilir, diğerlerini kendileriyle iletişime davet ederler.

Sosyal: Yuvarlak yüzlü bireyler sosyal ve dışa dönük olarak algılanabilirler, etkileşimleri ve ilişkileri keyifli bulabilirler.

İyiliksever: Yuvarlak bir yüz şekli nazik ve iyi niyetli bir doğayı ima edebilir, bakım ve düşünceli olma imajıyla uyum sağlar.

Mizahi: Bazıları yuvarlak bir yüzü mizah anlayışıyla ilişkilendirebilir, bu da kişinin gülmeyi ve neşeli olmayı sevdiğini ima edebilir.

Olumlu Tutum: Yüz yapısı, hayata olumlu ve iyimser bir bakış açısı ile iletebilir.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Good Guy' and res2 == 'Square':
    mesaj = f"""\
Probability of being Good Guy is {res}
Being {res2} face shape probability is {res3}

Kare yüz şekline sahip bir kişi genellikle güçlü, kendine güvenen ve kararlı olarak algılanır. "İyi insan" tanımı ile birleştiğinde, bu kişinin karakterinde bu olumlu özellikleri yansıttığını düşündürebilir. İşte bir "iyi insan" imajına ve kare yüze sahip bir kişiyle ilişkilendirebileceğiniz bazı ifadeler veya özellikler:

Güvenilir: Kare yüzlü insanlar genellikle güvenilirlik ve güvenilirlik izlenimi verir, bu da bir "iyi insan" olma fikriyle uyumlu bir özelliktir.

Güçlü Çene: Kare yüzler genellikle güçlü bir çeneyi simgeler, bu da kararlılık ve dayanıklılığı temsil eder.

Dürüst Görünüm: Kare yüz dürüstlüğü ve açıklığı ile ilişkilendirilebilir, bu da bir "iyi insan" fikrini pekiştirir.

Liderlik Potansiyeli: Kare yüzlü bireyler, yüz yapıları özgüven ve kararlılık gösterebileceğinden doğal liderler olarak görülebilirler.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Bad Guy' and res2 == 'Heart':
    mesaj = f"""\
Probability of being Bad Guy is {res}
Being {res2} face shape probability is {res3}

Kalp şeklinde bir yüze sahip bir "harikulade olmayan insan" karakterini tanımlarken, yüz hatlarının karakterlerine karmaşıklık kattığını söyleyebiliriz.

Onun kalp şeklindeki yüzünün yumuşak ve çekici hatlarına rağmen, onun hakkında ürkütücü bir şeyler var. Görünüşte masum özellikleri, içinde gizlenen kötülükle sert bir tezat oluşturuyor.
Sanki aldatıcı tatlı görünümünü, daha karanlık bir ajandayı gizlemek için bir maskara olarak kullanıyormuş gibi. Yüzünün nazik eğrilerine aldanmayın; bu görünüşün altında kurnaz ve hain bir zeka var.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Bad Guy' and res2 == 'Oblong':
    mesaj = f"""\
Probability of being Bad Guy is {res}
Being {res2} face shape probability is {res3}

Dikdörtgen yüzlü harikulade olmayan bir insanın yüz hatları uzun ve potansiyel olarak tehditkar görünebilir.

Onun dikdörtgen yüzü, genel görünümüne ürkütücü bir kalite katıyor. Uzun yüz hatları, tehditkar varlığını daha da artırıyor ve neredeyse başka bir dünyevi bir hava kazandırıyor.
Keskin, dar hatlara ve gözlerindeki tehditkar parıltıya sahip, manipülasyon ve aldatma konusunda zevk alan kötü adam tipini anımsatıyor. 
Dikdörtgen yüzü, her zaman kötü niyetli planlarında bir adım önde gibi göründüğü endişe duygusunu artırıyor gibi, sanki her zaman bir adım önde gibi.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Bad Guy' and res2 == 'Oval':
    mesaj = f"""\
Probability of being Bad Guy is {res}
Being {res2} face shape probability is {res3}

Oval yüzlü harikulade olmayan bir insan, daha esrarlı ve sofistike bir görünüme sahip olabilir.

Oval yüzü, hesaplanmış kötülüğün bir havasını yayar. Vücudu düzgün, simetrik hatları, gerilimi arttırmak için gerçek niyetini rahatsız edici bir cazibe ile örtüyor gibi görünüyor. 
Özellikleri, geleneksel olarak çekici olsa da, gizli bir ajandayı ima eden anlaşılmaz bir kalite taşıyor. Keskin bir bakış ve hesaplanmış ve zarif bir tavır ile, gölgede iş yapan, çekici oval yüzünü kullanarak karanlık ve sofistike doğasını gizlemek için kullanan harikulade olmayan bir karakter tipidir.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Bad Guy' and res2 == 'Round':
    mesaj = f"""\
Probability of being Bad Guy is {res}
Being {res2} face shape probability is {res3}

Yuvarlak yüzlü bir kötü karakter, insanları yanıltıcı bir görünüme sahip olabilir ve insanları yanıltıcı bir güven duygusuna sokabilir. İşte bir tanım:

Onun yuvarlak yüzü, görünüşte dostça cephesinin altında gizli bir kurnaz doğayı gizliyor. En dikkatli gözlemcileri bile etkisiz hale getirebilecek yumuşak, kavisli özelliklerle, kötülüğünü maskelemekte usta.
Yuvarlak yüzü, topluma sorunsuz bir şekilde karışmasına izin verirken, görünüşte masum tavırları yüzeyin altında pusuya düşen kötü niyetleri ele vermez. O, alçakgönüllü görünüşünü kullanarak manipüle etmek ve aldatmak için kullanan  harikulade olmayan bir insandır, bu da onu daha da tehlikeli kılar.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()
elif firstVal == 'Bad Guy' and res2 == 'Square':
    mesaj = f"""\
Probability of being Bad Guy is {res}
Being {res2} face shape probability is {res3}

Kare yüz şekli, harikulade olmayan bir insanın karakteri ile ilişkilendirildiğinde, etkileyici ve potansiyel olarak tehditkar bir görünümlerini vurgulayabilir.

Onun kare yüzü, hakimiyet ve tehditkarlık havası yayar. Keskin, oyma gibi köşeleri ve güçlü çene hattı ile, özellikleri göz ardı edilmesi zor bir hakim varlığı yansıtıyor gibi görünüyor.
Bu kare yüzlü kişi, etkileyici ve tehditkar bir kişilik taşıyormuş gibi görünüyor, onu yakından takip etmek isteyeceğiniz bir karakter olabilir.


Sosyal Medya Hesaplarımız:
https://linktr.ee/hacettepeaiclub
"""
    context = ssl.create_default_context()


#sending mail

posta = MIMEMultipart()
posta['From'] = user
posta['To'] = alici
posta['Subject'] = baslik

posta.attach(MIMEText(mesaj, 'plain'))
# Convert the message to a string
posta_str = posta.as_string()

"""eklenti_dosya_ismi = "sendMail.txt"
with(open(eklenti_dosya_ismi, 'rb')) as eklenti_dosyasi:
    payload = MIMEBase('application', "octate-stream")
    payload.set_payload((eklenti_dosyasi).read())
    encoders.encode_base64(payload)
    payload.add_header("Content-Decomposition", "attachment", filename = eklenti_dosya_ismi)
    posta.attach(payload)
    posta_str = posta.as_string()"""

port = 465
host = 'smtp.gmail.com'
context = ssl.create_default_context()

with smtplib.SMTP_SSL(host=host, port=port, context=context) as epostaSunucusu:
    epostaSunucusu.login(user, password)
    epostaSunucusu.sendmail(user, alici, posta_str)

"""epostaSunucusu = smtplib.SMTP_SSL(host = host, port = port, context= context)
epostaSunucusu.login(user, password)
epostaSunucusu.sendmail(user, alici, posta_str)"""