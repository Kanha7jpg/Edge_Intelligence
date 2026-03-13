import time
import threading
import numpy as np
import json
import struct
import socket
from network.p2p_node import P2PNode

def test_lamport_clocks():
    print("\n--- Testing Lamport Clocks ---")
    
    events = []
    def callback(event, data):
        if event == "CLOCK":
            events.append(data)
            print(f"[UI] Clock Updated: {data}")

    node_a = P2PNode("127.0.0.1", 7001, ui_callback=callback)
    node_b = P2PNode("127.0.0.1", 7002, ui_callback=callback)
    
    node_a.start()
    node_b.start()
    
    node_a.add_peer("127.0.0.1", 7002)
    node_b.add_peer("127.0.0.1", 7001)
    
    time.sleep(1) # wait for connection
    
    print("Node A sending message...")
    dummy_vector = np.zeros(512)
    node_a.broadcast_intelligence("test_1", dummy_vector, time.time())
    
    time.sleep(2) # wait for propagation
    
    print(f"Captured Clock Events: {events}")
    # Node A increments to 1 on send
    # Node B updates to max(0, 1) + 1 = 2 on receive
    assert 1 in events
    assert 2 in events
    
    node_a.stop()
    node_b.stop()
    print("--- Lamport Clock Test Passed ---")

def test_bully_election():
    print("\n--- Testing Bully Election ---")
    
    node_states = {7001: None, 7002: None, 7003: None}
    
    def get_callback(port):
        def cb(event, data):
            if event == "COORDINATOR":
                node_states[port] = data
                print(f"[Node {port}] Coordinator: {data}")
        return cb

    nodes = []
    for port in [7001, 7002, 7003]:
        n = P2PNode("127.0.0.1", port, ui_callback=get_callback(port))
        nodes.append(n)
        
    for n in nodes:
        for other_port in [7001, 7002, 7003]:
            if n.port != other_port:
                n.add_peer("127.0.0.1", other_port)
        n.start()

    print("Waiting for initial election...")
    time.sleep(5)
    
    # Highest port should be leader
    expected_leader = "127.0.0.1:7003"
    for port in node_states:
        print(f"Node {port} thinks leader is {node_states[port]}")
        if node_states[port] != expected_leader:
            print(f"FAIL: Node {port} has wrong leader {node_states[port]}")
        assert node_states[port] == expected_leader

    print("\nKilling leader (Node 7003)...")
    nodes[2].stop()
    
    # Trigger a BROADCAST from Node 7001 to detect the failure if needed, 
    # but the flush loop and PEER_DOWN should catch it eventually.
    # To speed it up, we can trigger a message.
    nodes[0].broadcast_intelligence("check_failure", np.zeros(512), time.time())
    
    print("Waiting for re-election...")
    time.sleep(10) # increase wait
    
    expected_new_leader = "127.0.0.1:7002"
    for port in [7001, 7002]:
        print(f"Node {port} now thinks leader is {node_states[port]}")
        if node_states[port] != expected_new_leader:
            print(f"FAIL: Node {port} after failure has leader {node_states[port]}")
        assert node_states[port] == expected_new_leader

    for n in nodes:
        n.stop()
    print("--- Bully Election Test Passed ---")

if __name__ == "__main__":
    try:
        test_lamport_clocks()
        test_bully_election()
        print("\n[SUCCESS] ALL DISTRIBUTED TESTS PASSED")
    except AssertionError as e:
        print(f"\n[FAILURE] TEST FAILED: {e}")
    except Exception as e:
        print(f"\n[ERROR] ERROR DURING TESTING: {e}")
