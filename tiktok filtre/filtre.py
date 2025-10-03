import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe modüllerini yükle
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Modelleri başlat
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2  # İki el takibi için
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kamerayı başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Çekme/uzatma parametreleri
grab_radius = 50
max_stretch = 1.8
pinch_threshold = 0.05
deformation_strength = 0.8

# Çekme durumu değişkenleri - her el için ayrı
is_grabbing = [False, False]  # İki el için durum
grab_start_x = [None, None]  # İki el için başlangıç X
grab_start_y = [None, None]  # İki el için başlangıç Y
last_frame = None
last_finger_pos = [None, None]  # İki el için son pozisyon
last_process_time = 0
process_every_n_frames = 2
frame_count = 0


def calculate_finger_distance(hand_landmarks, width):
    """Başparmak ve işaret parmağı arasındaki mesafeyi hesaplar"""
    thumb_x = hand_landmarks.landmark[4].x * width
    thumb_y = hand_landmarks.landmark[4].y
    index_x = hand_landmarks.landmark[8].x * width
    index_y = hand_landmarks.landmark[8].y

    distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
    return distance / width


def is_point_in_face(x, y, face_landmarks, width, height, margin=20):
    """Bir noktanın yüz bölgesi içinde olup olmadığını kontrol eder"""
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    key_points = [0, 10, 152, 234, 454]  # Yüz çevresindeki önemli noktalar

    for idx in key_points:
        landmark = face_landmarks.landmark[idx]
        px, py = int(landmark.x * width), int(landmark.y * height)
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px)
        max_y = max(max_y, py)

    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    return min_x <= x <= max_x and min_y <= y <= max_y


def optimized_deformation(image, start_x, start_y, current_x, current_y):
    """Optimize edilmiş deformasyon - orijinal fonksiyon korundu"""
    height, width = image.shape[:2]

    # Çekme vektörü ve mesafesi
    dx = current_x - start_x
    dy = current_y - start_y
    pull_distance = np.sqrt(dx * dx + dy * dy)

    if pull_distance < 3:
        return image

    # Çekme yönü
    direction_x = dx / pull_distance
    direction_y = dy / pull_distance

    # Uzama faktörü
    stretch_factor = min(1 + (pull_distance / 100) * deformation_strength, max_stretch)

    # Etkilenen bölgenin sınırlarını hesapla
    region_x1 = max(0, int(start_x - grab_radius - pull_distance))
    region_y1 = max(0, int(start_y - grab_radius - pull_distance))
    region_x2 = min(width, int(start_x + grab_radius + pull_distance))
    region_y2 = min(height, int(start_y + grab_radius + pull_distance))

    # Haritalama matrisleri
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Varsayılan olarak orijinal koordinatları ata
    for y in range(height):
        for x in range(width):
            map_x[y, x] = x
            map_y[y, x] = y

    # Sadece etkilenen bölgeyi işle
    step = 2
    for y in range(region_y1, region_y2, step):
        for x in range(region_x1, region_x2, step):
            # Başlangıç noktasına olan mesafe
            dx_point = x - start_x
            dy_point = y - start_y
            distance = np.sqrt(dx_point ** 2 + dy_point ** 2)

            if distance < grab_radius:
                # Ağırlık hesaplama
                weight = max(0, 1.0 - distance / grab_radius)

                # Deformasyon
                displacement = weight * (stretch_factor - 1) * pull_distance * deformation_strength

                # Yeni koordinatlar
                new_x = x - direction_x * displacement
                new_y = y - direction_y * displacement

                # Sınırları kontrol et
                new_x = max(0, min(width - 1, new_x))
                new_y = max(0, min(height - 1, new_y))

                # Adım boyutu için komşu pikselleri doldur
                for sy in range(step):
                    for sx in range(step):
                        if y + sy < region_y2 and x + sx < region_x2:
                            map_x[y + sy, x + sx] = new_x
                            map_y[y + sy, x + sx] = new_y

    # Deformasyonu uygula
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi çevir
    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Her zaman orijinal kareyi göster (eller tespit edilmese bile)
    display_frame = frame.copy()

    # İlk kare için referans kareyi ayarla
    if last_frame is None:
        last_frame = frame.copy()

    # Her kareyi işleme
    current_time = time.time()
    process_this_frame = frame_count % process_every_n_frames == 0

    # El ve yüz tespiti
    face_detected = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz tespiti
    face_results = face_mesh.process(rgb_frame)
    face_detected = face_results.multi_face_landmarks is not None

    # El tespiti
    hand_results = hands.process(rgb_frame)
    height, width, _ = frame.shape

    # Her el için değişkenleri sıfırla
    finger_x = [None, None]
    finger_y = [None, None]
    is_pinched = [False, False]

    if hand_results.multi_hand_landmarks:
        # En fazla 2 el için işlem yap
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if i >= 2:  # 2'den fazla el varsa atla
                break

            # El pozisyonunu ve kıstırma durumunu hesapla
            finger_x[i] = int(hand_landmarks.landmark[8].x * width)
            finger_y[i] = int(hand_landmarks.landmark[8].y * height)
            finger_distance = calculate_finger_distance(hand_landmarks, width)
            is_pinched[i] = finger_distance < pinch_threshold
            last_finger_pos[i] = (finger_x[i], finger_y[i])
    else:
        # Son bilinen parmak pozisyonlarını kullan
        for i in range(2):
            if last_finger_pos[i] is not None:
                finger_x[i], finger_y[i] = last_finger_pos[i]

    # Deformasyon işlemi
    apply_deformation = False

    # Her iki el için ayrı ayrı işlem yap
    if face_detected and face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]

        # Her el için deformasyon işlemi
        for i in range(2):
            # Eğer bu el için konum bilgisi yoksa atla
            if finger_x[i] is None or finger_y[i] is None:
                continue

            # Yüz üzerinde mi ve kıstırma var mı kontrol et
            if is_point_in_face(finger_x[i], finger_y[i], face_landmarks, width, height) and is_pinched[i]:
                if not is_grabbing[i]:
                    # İlk kez yakalama başlat
                    is_grabbing[i] = True
                    grab_start_x[i], grab_start_y[i] = finger_x[i], finger_y[i]
                    # İlk el için veya hiçbir el yakalamıyorsa referans kareyi güncelle
                    if i == 0 or not any(is_grabbing):
                        last_frame = frame.copy()
            elif is_grabbing[i] and is_pinched[i]:
                # Deformasyon uygula
                display_frame = optimized_deformation(
                    display_frame.copy(),  # Önceki elin deformasyonunu korumak için display_frame kullan
                    grab_start_x[i],
                    grab_start_y[i],
                    finger_x[i],
                    finger_y[i]
                )
                apply_deformation = True
            elif is_grabbing[i] and not is_pinched[i]:
                # Bırakma
                is_grabbing[i] = False

        # Hiçbir el yakalamıyorsa ve deformasyon uygulanmadıysa, normal kareyi göster
        if not any(is_grabbing) and not apply_deformation:
            display_frame = frame.copy()

    # Durum bilgisini göster (isteğe bağlı)
    active_hands = sum(1 for x in is_grabbing if x)
    cv2.putText(display_frame, f"Aktif eller: {active_hands}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Her durumda kareyi göster (eller tespit edilmese bile)
    cv2.imshow("Yüz Filtresi", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()