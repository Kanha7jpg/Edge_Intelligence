import cv2
import torch
import json
import time
import hashlib
import socket
import threading
import struct
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from queue import Queue
from ultralytics import YOLO

MY_NODE_ID = "NODE_A" # Changes per laptop
with open('config.json', 'r') as f: 
    TOPOLOGY = json.load(f)

MY_IP, MY_PORT = TOPOLOGY[MY_NODE_ID]["ip"], TOPOLOGY[MY_NODE_ID]["port"]
MY_NEIGHBORS = TOPOLOGY[MY_NODE_ID]["neighbors"]
SIMILARITY_THRESHOLD = TOPOLOGY.get("similarity_threshold", 0.75)
DEDUP_THRESHOLD = TOPOLOGY.get("dedup_threshold", 0.95)
MAX_MEMORY = TOPOLOGY.get("max_memory_entries", 500)
LOG_COOLDOWN = TOPOLOGY.get("log_cooldown_seconds", 5)

network_memory = []  # Stores all embeddings (local + received from neighbors)
memory_lock = threading.Lock()  # Thread-safe access to network_memory
opportunistic_queue = Queue()

person_counter = 0  # Auto-incrementing ID for new persons
counter_lock = threading.Lock()
last_log_times = {}  # Tracks last log timestamp per person_id

print("Loading YOLOv8...")
detector = YOLO('yolov8n.pt')

print("Loading ResNet-18...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

preprocess = T.Compose([
    T.ToPILImage(), 
    T.Resize((224, 224)), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_vector(img_crop):
    # Ensure image has 3 channels if it's grayscale
    if len(img_crop.shape) == 2:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
    
    img_tensor = preprocess(img_crop).unsqueeze(0)
    with torch.no_grad(): 
        return resnet(img_tensor).flatten().numpy()

def generate_commitment(vector):
    salt = str(time.time())
    commitment = hashlib.sha256(vector.tobytes() + salt.encode()).hexdigest()
    return commitment, salt

def verify_commitment(vector, commitment, salt):
    return hashlib.sha256(np.array(vector, dtype=np.float32).tobytes() + salt.encode()).hexdigest() == commitment


# --- Re-Identification Logic ---
def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def find_match(new_vector, threshold=SIMILARITY_THRESHOLD):
    """Search network_memory for the best matching embedding above threshold.
    Returns (matched_entry, similarity_score) or (None, 0.0)."""
    best_match = None
    best_score = -1.0
    
    with memory_lock:
        for entry in network_memory:
            score = cosine_similarity(new_vector, entry['vector'])
            if score > threshold and score > best_score:
                best_match = entry
                best_score = score
    
    return best_match, best_score

def update_or_append(entry, matched_entry=None):
    """Update an existing entry's vector (running average) or append a new one.
    Enforces MAX_MEMORY with LRU eviction (oldest entry removed first)."""
    with memory_lock:
        if matched_entry is not None:
            # Update existing entry with running average of embeddings
            old_vec = matched_entry['vector']
            new_vec = entry['vector']
            matched_entry['vector'] = 0.7 * old_vec + 0.3 * new_vec
        else:
            # New person — enforce memory cap before appending
            if len(network_memory) >= MAX_MEMORY:
                network_memory.pop(0)  # Evict oldest entry (LRU)
            network_memory.append(entry)

def should_log(person_id):
    """Rate-limit logs to at most once per LOG_COOLDOWN seconds per person."""
    now = time.time()
    if person_id not in last_log_times or (now - last_log_times[person_id]) >= LOG_COOLDOWN:
        last_log_times[person_id] = now
        return True
    return False

def get_next_person_id():
    """Thread-safe auto-incrementing person ID."""
    global person_counter
    with counter_lock:
        person_counter += 1
        return f"P-{person_counter}"


# --- Helper for exact byte retrieval ---
def recvall(sock, n):
    """Helper function to cleanly receive exactly n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


# --- Receiver Server Thread ---
def peer_to_peer_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows OS to reuse the port immediately if the Python script crashes/restarts
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('0.0.0.0', MY_PORT))
    server.listen(5)
    
    print(f"[{MY_NODE_ID}] Listening on Port {MY_PORT}...")
    
    while True:
        conn, addr = server.accept()
        try:
            # 1. Read EXACTLY 4 bytes to find out the payload size
            raw_msglen = recvall(conn, 4)
            if not raw_msglen:
                continue # Client closed connection abruptly
            
            # Unpack the 4 bytes back into a standard integer
            msglen = struct.unpack('>I', raw_msglen)[0]
            
            # 2. Read EXACTLY `msglen` bytes to get the full JSON payload
            message_data = recvall(conn, msglen)
            if not message_data:
                continue 
            
            # 3. Decode and parse
            payload = json.loads(message_data.decode('utf-8'))
            
            # 4. Enforce Vector Commitment Security
            if verify_commitment(payload['vector'], payload['commitment'], payload['salt']):
                received_vector = np.array(payload['vector'])
                payload['vector'] = received_vector
                
                # 5. Re-ID: Check if this person already exists in our memory
                match, score = find_match(received_vector)
                
                if match:
                    # Known person — update person_id and merge embedding
                    payload['person_id'] = match['person_id']
                    update_or_append(payload, matched_entry=match)
                    if should_log(match['person_id']):
                        print(f"[{MY_NODE_ID}] RE-ID MATCH: Person {match['person_id']} "
                              f"from {payload.get('origin', '?')} matched with "
                              f"existing entry from {match.get('origin', '?')} "
                              f"(similarity: {score:.3f})")
                else:
                    # New person we haven't seen before
                    if 'person_id' not in payload:
                        payload['person_id'] = get_next_person_id()
                    update_or_append(payload)
                    print(f"[{MY_NODE_ID}] + New person {payload['person_id']} "
                          f"synced from {payload.get('origin', 'Unknown')}")
                    
            else:
                print(f"[{MY_NODE_ID}] ! REJECTED: Invalid Commitment Hash from {addr}")
                
        except json.JSONDecodeError as e:
            print(f"[{MY_NODE_ID}] JSON parsing failed mid-stream: {e}")
        except Exception as e:
            print(f"[{MY_NODE_ID}] General socket error: {e}")
        finally:
            conn.close()


# --- Sender Function ---
def send_unicast(target_node, payload_dict):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((TOPOLOGY[target_node]["ip"], TOPOLOGY[target_node]["port"]))
        
        # 1. Serialize the dictionary to bytes
        message_data = json.dumps(payload_dict).encode('utf-8')
        
        # 2. Prevent fragmentation: Create a 4-byte length prefix header
        # '>I' means Big-Endian, Unsigned Integer (4 bytes)
        message_length = len(message_data)
        header = struct.pack('>I', message_length)
        
        # 3. Send the prefix FIRST, followed exactly by the payload
        s.sendall(header + message_data)
        s.close()
        return True
    except Exception as e:
        # Expected to fail if neighbor is offline
        return False

def opportunistic_network_worker():
    while True:
        if not opportunistic_queue.empty():
            item = opportunistic_queue.queue[0]
            if send_unicast(item['target'], item['payload']):
                # Only remove once successfully sent
                opportunistic_queue.get()
                print(f"[{MY_NODE_ID}] Opportunistically sent queued payload to {item['target']}")
        time.sleep(3)


if __name__ == "__main__":
    threading.Thread(target=peer_to_peer_server, daemon=True).start()
    threading.Thread(target=opportunistic_network_worker, daemon=True).start()

    # Track last sent embedding per person to avoid flooding
    last_sent_vectors = {}

    # Main Loop
    print(f"[{MY_NODE_ID}] Starting Camera...")
    print(f"[{MY_NODE_ID}] Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"[{MY_NODE_ID}] Dedup Threshold: {DEDUP_THRESHOLD}")
    print(f"[{MY_NODE_ID}] Max Memory Entries: {MAX_MEMORY}")
    print(f"[{MY_NODE_ID}] Log Cooldown: {LOG_COOLDOWN}s")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Failed to grab frame")
            break
        
        # YOLO Detection: only detect persons (class 0)
        results = detector(frame, classes=[0], verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop person
                img_crop = frame[y1:y2, x1:x2]
                
                if img_crop.size == 0:
                    continue
                    
                # Extract Vector
                current_vector = extract_vector(img_crop)
                
                # --- Re-ID: Match against network memory ---
                match, score = find_match(current_vector)
                
                if match:
                    # Known person — reuse their person_id and update embedding
                    person_id = match['person_id']
                    origin_node = match.get('origin', MY_NODE_ID)
                    
                    if origin_node != MY_NODE_ID:
                        label = f"{person_id} (from {origin_node}) [{score:.2f}]"
                        box_color = (0, 0, 255)  # Red for cross-node re-id
                        if should_log(person_id):
                            print(f"[{MY_NODE_ID}] RE-ID: {person_id} originally from {origin_node} (sim: {score:.3f})")
                    else:
                        label = f"{person_id} [{score:.2f}]"
                        box_color = (0, 255, 0)  # Green for locally known
                    
                    # Update existing entry's vector (running average)
                    local_update = {'vector': current_vector, 'origin': MY_NODE_ID, 'person_id': person_id}
                    update_or_append(local_update, matched_entry=match)
                else:
                    # New person — assign fresh ID
                    person_id = get_next_person_id()
                    label = f"{person_id} [NEW]"
                    box_color = (255, 165, 0)  # Orange for new person
                    print(f"[{MY_NODE_ID}] New person detected: {person_id}")
                    
                    # Store locally as new entry
                    local_entry = {'vector': current_vector, 'origin': MY_NODE_ID, 'person_id': person_id}
                    update_or_append(local_entry)
                
                # Generate Commitment
                commitment, salt = generate_commitment(current_vector)
                
                payload = {
                    "origin": MY_NODE_ID, 
                    "person_id": person_id,
                    "vector": current_vector.tolist(), 
                    "commitment": commitment, 
                    "salt": salt
                }
                
                # --- Deduplication: only send if sufficiently different ---
                should_send = True
                if person_id in last_sent_vectors:
                    dedup_score = cosine_similarity(current_vector, last_sent_vectors[person_id])
                    if dedup_score > DEDUP_THRESHOLD:
                        should_send = False  # Too similar to last sent, skip
                
                if should_send:
                    last_sent_vectors[person_id] = current_vector
                    for neighbor in MY_NEIGHBORS:
                        if not send_unicast(neighbor, payload):
                            # Queue for opportunistic networking if offline
                            opportunistic_queue.put({'target': neighbor, 'payload': payload})
                
                # Draw bounding box with re-id info
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
        # Show memory stats on frame
        with memory_lock:
            mem_count = len(network_memory)
        cv2.putText(frame, f"Memory: {mem_count} embeddings", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(f"{MY_NODE_ID} Camera view", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()