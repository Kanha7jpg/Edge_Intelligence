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

network_memory = []
opportunistic_queue = Queue()

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
    return hashlib.sha256(np.array(vector).tobytes() + salt.encode()).hexdigest() == commitment


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
                payload['vector'] = np.array(payload['vector'])
                network_memory.append(payload)
                print(f"[{MY_NODE_ID}] + Memory Synced from {payload.get('origin', 'Unknown')}")
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

    # Main Loop
    print(f"[{MY_NODE_ID}] Starting Camera...")
    cap = cv2.VideoCapture(0)
    
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
                
                # Generate Commitment
                commitment, salt = generate_commitment(current_vector)
                
                payload = {
                    "origin": MY_NODE_ID, 
                    "vector": current_vector.tolist(), 
                    "commitment": commitment, 
                    "salt": salt
                }
                
                # Send to neighbors
                for neighbor in MY_NEIGHBORS:
                    if not send_unicast(neighbor, payload):
                        # Queue for opportunistic networking if offline
                        opportunistic_queue.put({'target': neighbor, 'payload': payload})
                        
                # Draw bounding box for visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.imshow(f"{MY_NODE_ID} Camera view", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
