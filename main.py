import argparse
import threading
import time
import cv2
import uuid

from core.ml_engine import MLEngine
from core.feature_extractor import FeatureExtractor
from core.fusion import FeatureFusion
from core.tracker import ReIDTracker
from core.memory import LRUMemory
from network.p2p_node import P2PNode
from data.stimulus_loader import StimulusLoader
from ui.dashboard import NerveCenterUI

class CognitiveMeshNode:
    def __init__(self, host, port, stimulus_source):
        self.node_id = f"{host}:{port}"
        self.host = host
        self.port = port
        
        # Initialize Core Systems
        self.ml_engine = MLEngine()
        self.feature_extractor = FeatureExtractor()
        self.fusion_engine = FeatureFusion(alpha=0.3)
        self.tracker = ReIDTracker(similarity_threshold=0.85)
        self.memory = LRUMemory(max_size=100)
        self.stimulus_loader = StimulusLoader(stimulus_source)
        
        # Initialize UI First (needs to be main thread usually, but starting it last)
        self.ui = NerveCenterUI(
            node_id=self.node_id,
            on_start_sim=self.start_stimulus_simulation,
            on_add_peer=self.add_peer
        )
        
        # Initialize Network
        self.network = P2PNode(host, port, ui_callback=self.handle_network_event)
        
        self.running = False
        self.sim_thread = None

    def start(self):
        self.network.start()
        # UI mainloop blocks the main thread
        self.ui.mainloop()
        self.stop()

    def stop(self):
        self.running = False
        self.network.stop()

    def add_peer(self, host, port):
        self.network.add_peer(host, port)

    def handle_network_event(self, event_type, data):
        """ Callback from P2PNode when networking events occur """
        if event_type == "LOG":
            self.ui.log(data)
            
        elif event_type == "CONNECTION":
            self.ui.log(data)
            
        elif event_type == "PEER_UP":
            self.ui.log(f"Connected to peer {data}")
            
        elif event_type == "PEER_DOWN":
            self.ui.log(f"Lost connection to peer {data}")
            
        elif event_type == "PULSE":
            # Received intelligence from mesh
            global_id = data['global_id']
            incoming_vector = data['vector']
            
            # Fuse with local memory if exists
            local_data = self.memory.get(global_id)
            if local_data is not None:
                fused_vector = self.fusion_engine.fuse(local_data['features'], incoming_vector)
                self.memory.put(global_id, fused_vector, time.time())
                self.ui.log(f"FUSED: {global_id[:8]}")
            else:
                self.memory.put(global_id, incoming_vector, time.time())
                self.ui.log(f"NEW DB: {global_id[:8]}")

        elif event_type == "CLOCK":
            self.ui.update_clock(data)
            
        elif event_type == "COORDINATOR":
            self.ui.update_coordinator(data)
            
        elif event_type == "QUEUE_STATUS":
            self.ui.update_queues(data)

    def start_stimulus_simulation(self):
        if self.sim_thread and self.sim_thread.is_alive():
            self.ui.log("Simulation already running.")
            return
            
        self.running = True
        self.sim_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self.sim_thread.start()
        self.ui.log("Started Stimulus Simulation.")

    def _run_inference_loop(self):
        while self.running:
            frame = self.stimulus_loader.get_next_frame()
            if frame is None:
                self.ui.log("Stimulus sequence finished. Looping...")
                self.stimulus_loader.reset()
                continue
                
            # Process Frame (Local Intelligence)
            detections = self.ml_engine.process_frame(frame)
            
            for (x1, y1, x2, y2, conf) in detections:
                # Extact padded crop
                crop = self.ml_engine.get_padded_crop(frame, (x1, y1, x2, y2, conf))
                if crop is None or crop.size == 0:
                    continue
                    
                # Extract 512-D Feature Ghost
                feature_vector = self.feature_extractor.extract_features(crop)
                if feature_vector is None:
                    continue
                    
                # Try to Re-ID locally
                match_id, similarity = self.tracker.match(feature_vector, self.memory)
                
                if match_id is not None:
                    # Known identity - Fuse & Update
                    self.ui.log_match(match_id, similarity)
                    local_data = self.memory.get(match_id)
                    fused_vector = self.fusion_engine.fuse(local_data['features'], feature_vector)
                    self.memory.put(match_id, fused_vector, time.time())
                    
                    # Draw rect (Red for known/verified from mesh)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{match_id[:8]} ({similarity:.2f})"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Broadcast updated intelligence
                    self.network.broadcast_intelligence(match_id, fused_vector, time.time())
                    
                else:
                    # New Identity
                    new_id = str(uuid.uuid4())
                    self.memory.put(new_id, feature_vector, time.time())
                    self.ui.log(f"New Identity Captured: {new_id[:8]}")
                    
                    # Draw rect (Green for new)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, new_id[:8], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Broadcast new intelligence
                    self.network.broadcast_intelligence(new_id, feature_vector, time.time())
            
            # Update UI Video Feed
            self.ui.update_frame(frame)
            time.sleep(0.03) # simulate normal fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--source", default="0", help="Path to video or image directory")
    args = parser.parse_args()
    
    node = CognitiveMeshNode(args.host, args.port, args.source)
    node.start()
