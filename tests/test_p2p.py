import time
import threading
import numpy as np

from network.p2p_node import P2PNode
from network.message_queue import ThreadSafeQueue

def mock_ui_callback(event, data):
    pass # print(f"[{event}] {data}")

def test_store_and_forward():
    print("--- Testing Store and Forward ---")
    
    # Start Node A (Sender)
    node_a = P2PNode("127.0.0.1", 6001, ui_callback=mock_ui_callback)
    node_a.start()
    
    # Add peer B (Not currently running)
    node_a.add_peer("127.0.0.1", 6002)
    
    # Broadcast while B is offline -> Should queue
    dummy_vector = np.random.rand(512)
    node_a.broadcast_intelligence("test_id_1", dummy_vector, time.time())
    
    # Wait a bit
    time.sleep(1)
    
    with node_a.peer_lock:
        qsize = node_a.peers["127.0.0.1:6002"]["queue"].qsize()
        print(f"Queue size while B offline: {qsize} (Expected: 1)")
        assert qsize == 1
        
    print("Starting Node B to trigger flush...")
    node_b = P2PNode("127.0.0.1", 6002, ui_callback=mock_ui_callback)
    node_b.start()
    
    # Give opportunistic queue time to connect and flush
    time.sleep(3)
    
    with node_a.peer_lock:
        qsize = node_a.peers["127.0.0.1:6002"]["queue"].qsize()
        print(f"Queue size after B online: {qsize} (Expected: 0)")
        assert qsize == 0
        
    node_a.stop()
    node_b.stop()
    print("--- Store and Forward Test Passed ---\n")

def test_fusion_math():
    print("--- Testing Fusion Math ---")
    from core.fusion import FeatureFusion
    
    fusion = FeatureFusion(alpha=0.5)
    
    vec1 = np.ones(512)
    vec1 = vec1 / np.linalg.norm(vec1)
    
    vec2 = np.zeros(512)
    vec2[0] = 1.0 # purely one direction
    
    fused = fusion.fuse(vec1, vec2)
    
    norm = np.linalg.norm(fused)
    print(f"Fused vector norm: {norm:.4f} (Expected: 1.0)")
    assert np.isclose(norm, 1.0)
    
    print("--- Fusion Math Test Passed ---\n")


if __name__ == "__main__":
    test_fusion_math()
    test_store_and_forward()
    print("All tests passed.")
