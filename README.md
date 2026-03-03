# Distributed Multi-Camera Tracking using Collaborative Edge Feature Fusion

## Overview

Welcome to the **Edge Co-Intelligence** application! This decentralized, peer-to-peer (P2P) distributed multi-camera tracking system enables multiple independent edge nodes (laptops, Jetson Nanos, etc.) to securely collaborate and track individuals across camera blind spots without relying on any centralized cloud infrastructure or message brokers (like MQTT or Kafka).

This ensures real-time processing, zero external dependencies, robust data privacy, and extreme fault tolerance.

---

## Architecture & Concepts

This project is built upon five foundational pillars designed to satisfy strict privacy and performance constraints:

### 1. Edge Co-Intelligence (Local Processing Only)
Instead of streaming raw, bandwidth-heavy video feeds to a centralized cloud for processing, all inferences are run locally on the edge nodes.
- **Object Detection**: Uses `YOLOv8n` to accurately detect individuals in the camera frame.
- **Feature Extraction**: Uses `ResNet-18` (with the final classification layer removed) to mathematically describe the detected person, converting a cropped image of the individual into a rich 512-dimensional numerical feature vector.
These numerical vectors (not images) are what the nodes share with each other to track individuals.

### 2. Topology-Aware Routing (No Broadcasts, No Brokers)
This system operates on a **100% decentralized P2P architecture**. There is no central MQTT broker or server. 
- Nodes only communicate directly via **raw TCP Sockets**.
- Communication is strictly **Unicast** based on an explicit `config.json` file. Each node only opens TCP connections to its predefined physical "neighbor" nodes, completely avoiding noisy network-wide broadcasts.
- Robust packet delivery is ensured through **Length-Prefixed Framing**, preventing JSON fragmentation over TCP streams by sending exactly 4 bytes denoting payload length before the payload itself.

### 3. Privacy via Vector Commitments (Zero Trust Sharing)
To maintain absolute privacy across the network, nodes **do not trust raw feature vectors immediately**.
- Before sending a vector, the sending node generates a **Cryptographic Commitment** by combining the 512-D vector byte stream with a randomized timestamp (salt), and hashing them together using `SHA-256`.
- The sending node transmits the vector, the salt, and the `SHA-256` commitment.
- Upon receiving a payload, the receiving node mathematically verifies the payload by hashing the received vector and salt. If the resulting hash matches the commitment, the vector is verified as untampered and is safely saved into the node's local network memory.

### 4. Opportunistic Networks (Zero Data Loss via Queuing)
Edge environments are notoriously unstable (Wi-Fi dropouts, nodes rebooting). This system guarantees high fault-tolerance through opportunistic data delivery.
- If a target node temporarily goes offline, the `send_unicast` function silently fails.
- Instead of dropping the tracking data, the sending node pushes the payload into a thread-safe local `Queue()`.
- A dedicated background `opportunistic_network_worker` thread continuously attempts to flush this queue, holding onto the data until the target neighbor comes back online, ensuring 100% eventual delivery.

---

## Setup & Running the 4-Laptop Test

### Requirements
You will need Python 3.8+ installed. Install the dependencies using the included `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Steps to Run
1. Ensure all four nodes (Node A, Node B, Node C, and Node D) are connected to the same local network (Wi-Fi).
2. Obtain the IPv4 address of all four laptops.
3. Update `config.json` with the exact IP addresses of all four laptops:
```json
{
    "NODE_A": {
        "ip": "192.168.1.10", 
        "port": 5000, 
        "neighbors": ["NODE_B", "NODE_C", "NODE_D"]
    },
    "NODE_B": {
        "ip": "192.168.1.11", 
        "port": 5000, 
        "neighbors": ["NODE_A", "NODE_C", "NODE_D"]
    },
    "NODE_C": {
        "ip": "192.168.1.12", 
        "port": 5000, 
        "neighbors": ["NODE_A", "NODE_B", "NODE_D"]
    },
    "NODE_D": {
        "ip": "192.168.1.13", 
        "port": 5000, 
        "neighbors": ["NODE_A", "NODE_B", "NODE_C"]
    }
}
```
4. On Laptop 1 (NODE_A):
   - Edit `core_node.py` and ensure `MY_NODE_ID = "NODE_A"`.
   - Run the script: `python core_node.py`.
5. On Laptop 2 (NODE_B):
   - Edit `core_node.py` and ensure `MY_NODE_ID = "NODE_B"`.
   - Run the script: `python core_node.py`.
6. On Laptop 3 (NODE_C):
   - Edit `core_node.py` and ensure `MY_NODE_ID = "NODE_C"`.
   - Run the script: `python core_node.py`.
7. On Laptop 4 (NODE_D):
   - Edit `core_node.py` and ensure `MY_NODE_ID = "NODE_D"`.
   - Run the script: `python core_node.py`.

Once running, stand in front of one camera. The YOLOv8 model will detect you, ResNet will extract your 512-D feature vector, encrypt it with SHA-256 constraints, and send it to the other three nodes. If any node goes offline temporarily, the sending node will securely buffer the payloads in its Opportunistic Queue until the offline node returns!
