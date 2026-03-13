import threading
import time

class BullyElection:
    def __init__(self, host, port, peers, send_callback, ui_callback):
        self.host = host
        self.port = port
        self.peers = peers # Reference to node.peers
        self.send_callback = send_callback # function(host, port, payload)
        self.ui_callback = ui_callback
        
        self.coordinator = None
        self.is_election_ongoing = False
        self.ok_received = False
        self.lock = threading.Lock()

    def start_election(self):
        """ Initiate the Bully election process. """
        with self.lock:
            if self.is_election_ongoing:
                return
            self.is_election_ongoing = True
            self.ok_received = False

        self.ui_callback("LOG", f"Node {self.port} starting election...")
        
        higher_nodes = []
        for peer_id, info in self.peers.items():
            if info['port'] > self.port:
                higher_nodes.append(info)
        
        if not higher_nodes:
            self.declare_victory()
            with self.lock:
                self.is_election_ongoing = False
        else:
            for node in higher_nodes:
                self.send_callback(node['host'], node['port'], {
                    "type": "ELECTION",
                    "sender_port": self.port,
                    "sender_host": self.host
                })
            
            # Use a timer to wait for any OK message
            threading.Timer(3.0, self._check_election_results).start()

    def _check_election_results(self):
        with self.lock:
            if not self.ok_received:
                self.declare_victory()
            self.is_election_ongoing = False

    def handle_message(self, payload):
        msg_type = payload.get("type")
        sender_port = payload.get("sender_port")
        sender_host = payload.get("sender_host")

        if msg_type == "ELECTION":
            if sender_port < self.port:
                # Send OK back and start own election
                self.send_callback(sender_host, sender_port, {
                    "type": "OK",
                    "sender_port": self.port,
                    "sender_host": self.host
                })
                self.start_election()
        
        elif msg_type == "OK":
            with self.lock:
                self.ok_received = True
        
        elif msg_type == "VICTORY":
            self.coordinator = f"{sender_host}:{sender_port}"
            self.ui_callback("COORDINATOR", self.coordinator)
            self.ui_callback("LOG", f"New Coordinator elected: {self.coordinator}")
            with self.lock:
                self.is_election_ongoing = False

    def declare_victory(self):
        self.coordinator = f"{self.host}:{self.port}"
        self.ui_callback("COORDINATOR", self.coordinator)
        self.ui_callback("LOG", f"ELECTION WON: Node {self.port} is Leader")
        
        for peer_id, info in self.peers.items():
            self.send_callback(info['host'], info['port'], {
                "type": "VICTORY",
                "sender_port": self.port,
                "sender_host": self.host
            })
