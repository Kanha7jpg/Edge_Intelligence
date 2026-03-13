import customtkinter as ctk
import cv2
from PIL import Image
import threading
import time
import queue

class NerveCenterUI(ctk.CTk):
    def __init__(self, node_id, on_start_sim, on_add_peer):
        super().__init__()
        
        self.title(f"Nerve Center - Node {node_id}")
        self.geometry("1000x800")
        
        # Callbacks
        self.on_start_sim = on_start_sim
        self.on_add_peer = on_add_peer
        
        self._build_ui()
        self.log_messages = []
        
        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self._process_ui_queue()
        
    def _process_ui_queue(self):
        """Processes all pending UI updates in the main thread."""
        try:
            while True:
                task = self.ui_queue.get_nowait()
                func = task[0]
                args = task[1:]
                func(*args)
        except:
            pass
        self.after(50, self._process_ui_queue)
        
    def _build_ui(self):
        # Top panel: Controls
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=10, fill="x")
        
        self.peer_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Peer IP:Port (e.g., 127.0.0.1:5002)")
        self.peer_entry.pack(side="left", padx=10)
        
        self.add_peer_btn = ctk.CTkButton(self.control_frame, text="Add Peer", command=self._handle_add_peer)
        self.add_peer_btn.pack(side="left", padx=10)
        
        self.sim_btn = ctk.CTkButton(self.control_frame, text="Start Stimulus", command=self.on_start_sim)
        self.sim_btn.pack(side="right", padx=10)
        
        # Status Bar: Clock & Coordinator
        self.status_bar = ctk.CTkFrame(self, height=40, fg_color="transparent")
        self.status_bar.pack(pady=5, padx=10, fill="x")
        
        self.clock_label = ctk.CTkLabel(self.status_bar, text="Lamport Clock: 0", font=("Arial", 14, "bold"), text_color="#00FF00")
        self.clock_label.pack(side="left", padx=20)
        
        self.coord_label = ctk.CTkLabel(self.status_bar, text="Coordinator: Unknown", font=("Arial", 14), text_color="#3399FF")
        self.coord_label.pack(side="right", padx=20)
        
        # Middle panel: Video & Matches
        self.middle_frame = ctk.CTkFrame(self)
        self.middle_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.video_label = ctk.CTkLabel(self.middle_frame, text="Local Stimulus View")
        self.video_label.pack(side="left", expand=True, fill="both")
        
        self.match_frame = ctk.CTkFrame(self.middle_frame, width=300)
        self.match_frame.pack(side="right", fill="y", padx=10)
        
        self.match_label = ctk.CTkLabel(self.match_frame, text="Re-ID Match Gallery", font=("Arial", 16, "bold"))
        self.match_label.pack(pady=10)
        
        self.match_text = ctk.CTkTextbox(self.match_frame, width=280, height=300)
        self.match_text.pack(pady=10)
        
        # Bottom panel: Network Health & Logs
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(pady=10, padx=10, fill="x")
        
        self.log_text = ctk.CTkTextbox(self.bottom_frame, height=150)
        self.log_text.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.queue_frame = ctk.CTkFrame(self.bottom_frame, width=200)
        self.queue_frame.pack(side="right", fill="y", padx=10, pady=10)
        
        self.queue_label = ctk.CTkLabel(self.queue_frame, text="Peer Queues", font=("Arial", 12, "bold"))
        self.queue_label.pack(pady=5)
        
        self.queue_text = ctk.CTkTextbox(self.queue_frame, width=180, height=100, font=("Consolas", 10))
        self.queue_text.pack(pady=5)
        
    def _handle_add_peer(self):
        peer_addr = self.peer_entry.get().strip()
        if peer_addr:
            try:
                if ":" in peer_addr:
                    ip, port_str = peer_addr.split(":", 1)
                    port = int(port_str)
                else:
                    ip = peer_addr
                    port = 5001 # Default port
                    
                self.on_add_peer(ip, port)
                self.log(f"Added peer target {ip}:{port}")
                self.peer_entry.delete(0, 'end')
            except Exception as e:
                self.log(f"Error adding peer: {e}")

    def update_frame(self, frame_bgr):
        self.ui_queue.put((self._do_update_frame, frame_bgr))

    def _do_update_frame(self, frame_bgr):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        # Resize to fit UI
        image.thumbnail((500, 500))
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.video_label.configure(image=ctk_image, text="")
        
    def log(self, msg):
        self.ui_queue.put((self._do_log, msg))

    def _do_log(self, msg):
        self.log_messages.append(msg)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
        
        # Update text box safely
        self.log_text.delete("0.0", "end")
        self.log_text.insert("0.0", "\n".join(self.log_messages))
        self.log_text.see("end")
        
    def log_match(self, global_id, similarity):
        self.ui_queue.put((self._do_log_match, global_id, similarity))

    def _do_log_match(self, global_id, similarity):
        msg = f"Match! ID: {global_id[:8]} | Sim: {similarity:.2f}"
        self.match_text.insert("end", msg + "\n")
        self.match_text.see("end")

    def update_clock(self, ts):
        self.ui_queue.put((self._do_update_clock, ts))

    def _do_update_clock(self, ts):
        self.clock_label.configure(text=f"Lamport Clock: {ts}")

    def update_coordinator(self, coord):
        self.ui_queue.put((self._do_update_coordinator, coord))

    def _do_update_coordinator(self, coord):
        self.coord_label.configure(text=f"Coordinator: {coord}")

    def update_queues(self, status):
        self.ui_queue.put((self._do_update_queues, status))

    def _do_update_queues(self, status):
        self.queue_text.delete("0.0", "end")
        for peer, count in status.items():
            self.queue_text.insert("end", f"{peer}: {count}\n")
