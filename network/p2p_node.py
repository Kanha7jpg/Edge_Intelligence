import socket
import threading
import json
import struct
import time
import numpy as np

from network.message_queue import ThreadSafeQueue
from network.security import VectorCommitment
from network.election import BullyElection

class P2PNode:
    def __init__(self, host, port, ui_callback=None):
        self.host = host
        self.port = port
        self.ui_callback = ui_callback
        
        # Peers: { 'ip:port': {'queue': ThreadSafeQueue(), 'connected': False} }
        self.peers = {} 
        self.peer_lock = threading.Lock()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.running = False
        self.server_thread = None
        self.flush_thread = None
        self.lamport_clock = 0
        
        self.coordinator = None
        self.election_engine = BullyElection(
            host=self.host, 
            port=self.port, 
            peers=self.peers, 
            send_callback=self._send_election_msg,
            ui_callback=self.ui_callback
        )

    def add_peer(self, peer_host, peer_port):
        peer_port = int(peer_port)
        peer_id = f"{peer_host}:{peer_port}"
        with self.peer_lock:
            if peer_id not in self.peers:
                self.peers[peer_id] = {
                    'host': peer_host,
                    'port': peer_port,
                    'queue': ThreadSafeQueue(),
                    'connected': False,
                    'last_try': 0
                }
        
        # Re-evaluate leadership now that the peer set has changed
        self.election_engine.start_election()

    def start(self):
        self.running = True
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        self.flush_thread = threading.Thread(target=self._queue_flush_loop, daemon=True)
        self.flush_thread.start()
        
        if self.ui_callback:
            self.ui_callback("LOG", f"Node started on {self.host}:{self.port}")
        
        # Start initial election
        threading.Timer(2.0, self.election_engine.start_election).start()

    def stop(self):
        self.running = False
        self.server_socket.close()

    def _server_loop(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except OSError:
                break

    def _recv_all(self, conn, n):
        """ Helper function to perfectly read n bytes (solves fragmentation) """
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def _handle_client(self, conn, addr):
        if self.ui_callback:
            self.ui_callback("CONNECTION", f"Incoming connection from {addr}")
            
        while self.running:
            try:
                # Read 4-byte message length
                raw_msglen = self._recv_all(conn, 4)
                if not raw_msglen:
                    break
                msglen = struct.unpack('>I', raw_msglen)[0]
                
                # Read the actual message
                msg_data = self._recv_all(conn, msglen)
                if not msg_data:
                    break
                    
                # Decode JSON
                payload = json.loads(msg_data.decode('utf-8'))
                
                # Fix distributed routing by resolving 0.0.0.0 to the actual socket IP
                if payload.get("sender_host") == "0.0.0.0":
                    payload["sender_host"] = addr[0]
                
                # Normalize localhost for consistency
                if payload.get("sender_host") in ["::1", "localhost"]:
                    payload["sender_host"] = "127.0.0.1"
                    
                self._process_incoming_payload(payload)
                
            except Exception as e:
                break
        conn.close()

    def _process_incoming_payload(self, payload):
        # Passive Discovery: Add sender to peers if not already known
        sender_host = payload.get("sender_host")
        sender_port = payload.get("sender_port")
        if sender_host and sender_port:
            peer_id = f"{sender_host}:{sender_port}"
            # Don't add ourselves as a peer
            if peer_id != f"{self.host}:{self.port}":
                with self.peer_lock:
                    if peer_id not in self.peers:
                        # Use a thread so we don't block the receiver loop
                        threading.Thread(target=self.add_peer, args=(sender_host, sender_port), daemon=True).start()

        if "type" in payload and payload["type"] in ["ELECTION", "OK", "VICTORY"]:
            self.election_engine.handle_message(payload)
            return

        # Verify SHA-256
        global_id = payload.get("global_id")
        timestamp = payload.get("timestamp")
        vector_list = payload.get("vector")
        provided_hash = payload.get("hash")
        
        vector = np.array(vector_list)
        is_valid = VectorCommitment.verify_commitment(vector, global_id, timestamp, provided_hash)
        
        if is_valid and self.ui_callback:
            received_ts = payload.get("lamport_ts", 0)
            self.lamport_clock = max(self.lamport_clock, received_ts) + 1
            self.ui_callback("CLOCK", self.lamport_clock)
            self.ui_callback("PULSE", {"global_id": global_id, "vector": vector, "valid": True})
        elif not is_valid and self.ui_callback:
            self.ui_callback("LOG", f"Tampering detected! Invalid hash for {global_id}")

    def broadcast_intelligence(self, global_id, vector, timestamp):
        """
        Called when local ML extracts a feature ghost.
        Broadcasts to all known peers.
        """
        self.lamport_clock += 1
        if self.ui_callback:
            self.ui_callback("CLOCK", self.lamport_clock)

        payload = {
            "global_id": global_id,
            "timestamp": timestamp,
            "lamport_ts": self.lamport_clock,
            "sender_host": self.host,
            "sender_port": self.port,
            "vector": vector.tolist(),
            "hash": VectorCommitment.generate_commitment(vector, global_id, timestamp)
        }
        
        msg_bytes = json.dumps(payload).encode('utf-8')
        # Structure: 4-byte length prefix + actual bytes
        framed_msg = struct.pack('>I', len(msg_bytes)) + msg_bytes
        
        with self.peer_lock:
            for peer_id, peer_info in self.peers.items():
                peer_info['queue'].enqueue(framed_msg)

    def _queue_flush_loop(self):
        """
        Opportunistic Store-and-Forward:
        Periodically checks queues and attempts to flush them to peers.
        """
        while self.running:
            queue_status = {}
            with self.peer_lock:
                peer_items = list(self.peers.items())
                
            for peer_id, peer_info in peer_items:
                queue_status[peer_id] = peer_info['queue'].qsize()
                if not peer_info['queue'].is_empty():
                    # Attempt connection if not connected (debounced)
                    now = time.time()
                    if not peer_info['connected'] and now - peer_info['last_try'] > 2.0:
                        peer_info['last_try'] = now
                        success, s = self._attempt_connect(peer_info['host'], peer_info['port'])
                        if success:
                            peer_info['connected'] = True
                            peer_info['socket'] = s
                            if self.ui_callback:
                                self.ui_callback("PEER_UP", peer_id)
                        else:
                            if self.ui_callback:
                                self.ui_callback("PEER_DOWN", peer_id)
                            # If coordinator is unreachable, start election
                            if self.election_engine.coordinator == peer_id:
                                self.election_engine.start_election()
                                
                    # If connected, flush queue
                    if peer_info['connected']:
                        messages = peer_info['queue'].dequeue_all()
                        try:
                            for msg in messages:
                                peer_info['socket'].sendall(msg)
                        except Exception as e:
                            # Connection lost, put remaining back in queue
                            peer_info['connected'] = False
                            peer_info['socket'].close()
                            if self.ui_callback:
                                self.ui_callback("PEER_DOWN", peer_id)
                            
                            # If coordinator went down, start election
                            if self.election_engine.coordinator == peer_id:
                                self.election_engine.start_election()
                                
                            for msg in messages:
                                peer_info['queue'].enqueue(msg) # Re-queue
            
            if self.ui_callback:
                self.ui_callback("QUEUE_STATUS", queue_status)
                                
            time.sleep(1) # check queues every second

    def _send_election_msg(self, host, port, payload):
        """ Direct message sending for election control (bypasses main queue if needed) """
        try:
            msg_bytes = json.dumps(payload).encode('utf-8')
            framed_msg = struct.pack('>I', len(msg_bytes)) + msg_bytes
            
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((host, port))
            s.sendall(framed_msg)
            s.close()
        except Exception as e:
            if self.ui_callback:
                self.ui_callback("LOG", f"Election msg failed to {host}:{port}: {e}")

    def _attempt_connect(self, host, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((host, port))
            s.settimeout(None) # Make blocking again for long-lived usage
            return True, s
        except Exception:
            return False, None
