import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tech_support_agent import TechSupportAgent
import threading
from datetime import datetime

class TechSupportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Support Assistant")
        self.root.geometry("1400x900")
        
        # Set theme colors
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#1976D2',
            'background': '#F5F5F5',
            'text': '#333333',
            'success': '#4CAF50',
            'warning': '#FFC107',
            'chat_bg': '#FFFFFF',
            'user_msg': '#E3F2FD',
            'ai_msg': '#F5F5F5'
        }
        
        # Initialize the tech support agent
        self.agent = TechSupportAgent()
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure("TFrame", background=self.colors['background'])
        self.style.configure("TLabel", 
                           background=self.colors['background'],
                           foreground=self.colors['text'],
                           font=('Segoe UI', 10))
        self.style.configure("Title.TLabel",
                           font=('Segoe UI', 20, 'bold'),
                           foreground=self.colors['primary'])
        self.style.configure("TButton",
                           font=('Segoe UI', 10),
                           background=self.colors['primary'],
                           foreground='white')
        self.style.map("TButton",
                      background=[('active', self.colors['secondary'])])
        self.style.configure("Chat.TButton",
                           font=('Segoe UI', 10, 'bold'),
                           padding=5)
        self.style.configure("TLabelframe", 
                           background=self.colors['background'],
                           foreground=self.colors['text'])
        self.style.configure("TLabelframe.Label", 
                           font=('Segoe UI', 10, 'bold'),
                           foreground=self.colors['primary'])
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.chat_tab = ttk.Frame(self.notebook)
        self.kb_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.chat_tab, text="AI Chat")
        self.notebook.add(self.kb_tab, text="Knowledge Base")
        
        # Create and place widgets
        self.create_chat_widgets()
        self.create_kb_widgets()
        
        # Initialize chat history
        self.chat_history = []

    def create_chat_widgets(self):
        # Chat Frame
        chat_frame = ttk.Frame(self.chat_tab, padding="10")
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=('Segoe UI', 11),
            background=self.colors['chat_bg'],
            foreground=self.colors['text']
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        
        # Input Frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        # Message Entry
        self.message_entry = ttk.Entry(
            input_frame,
            font=('Segoe UI', 11)
        )
        self.message_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.message_entry.bind("<Return>", lambda e: self.send_message())
        
        # Send Button
        send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            style="Chat.TButton"
        )
        send_button.grid(row=0, column=1)

    def create_kb_widgets(self):
        # KB Frame
        kb_frame = ttk.Frame(self.kb_tab, padding="10")
        kb_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        kb_frame.columnconfigure(0, weight=1)
        kb_frame.rowconfigure(1, weight=1)
        
        # Search Frame
        search_frame = ttk.Frame(kb_frame)
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        # Search Label
        search_label = ttk.Label(
            search_frame,
            text="Search Knowledge Base:",
            font=('Segoe UI', 11)
        )
        search_label.grid(row=0, column=0, padx=(0, 10))
        
        # Search Entry
        self.search_entry = ttk.Entry(
            search_frame,
            font=('Segoe UI', 11)
        )
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.search_kb())
        
        # Search Button
        search_button = ttk.Button(
            search_frame,
            text="Search",
            command=self.search_kb,
            style="Chat.TButton"
        )
        search_button.grid(row=0, column=2)
        
        # Results Frame
        results_frame = ttk.Frame(kb_frame)
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Replace Listbox with Text widget for better formatting
        self.results_display = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            height=25,
            background='white',
            foreground=self.colors['text'],
            cursor="hand2"
        )
        self.results_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        self.results_display.config(state=tk.DISABLED)
        
        # Bind click events
        self.results_display.bind("<Button-1>", self.on_result_click)
        
        # Store results data for click handling
        self.results_data = []
        
        # Solution Display
        self.solution_display = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=80,
            height=35,
            font=('Segoe UI', 11),
            background='white',
            foreground=self.colors['text']
        )
        self.solution_display.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.solution_display.config(state=tk.DISABLED)

    def send_message(self):
        message = self.message_entry.get().strip()
        if not message:
            return
        
        # Clear entry
        self.message_entry.delete(0, tk.END)
        
        # Add user message to chat
        self.add_chat_message("You", message, is_user=True)
        
        # Process message in a separate thread
        def process():
            try:
                # Get AI response
                rag_response = self.agent.qa_chain({"query": message})
                
                # Get similar solutions with preview
                similar_solutions = self.agent.find_similar_solutions_with_preview(message, threshold=0.30)
                
                # Prepare AI response
                ai_response = rag_response["result"]
                
                if similar_solutions:
                    ai_response += "\n\nRelevant knowledge base articles found. Check the Knowledge Base tab for details."
                
                # Add AI response to chat
                self.root.after(0, lambda: self.add_chat_message("AI", ai_response, is_user=False))
                
                # Update KB tab if solutions found
                if similar_solutions:
                    self.root.after(0, lambda: self.update_kb_results_with_preview(similar_solutions))
            
            except Exception as e:
                error_message = f"Error: {str(e)}\nPlease try rephrasing your message or check if Ollama is running properly."
                self.root.after(0, lambda: self.add_chat_message("System", error_message, is_user=False))
        
        threading.Thread(target=process, daemon=True).start()

    def add_chat_message(self, sender, message, is_user=False):
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add message with formatting
        self.chat_display.insert(tk.END, f"\n{timestamp} - {sender}:\n", "sender")
        self.chat_display.insert(tk.END, f"{message}\n", "message")
        
        # Configure tags
        self.chat_display.tag_config("sender", font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config("message", 
                                   background=self.colors['user_msg'] if is_user else self.colors['ai_msg'],
                                   spacing1=5, spacing3=5)
        
        # Scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Store in history
        self.chat_history.append((sender, message, is_user))

    def search_kb(self):
        query = self.search_entry.get().strip()
        if not query:
            return
        
        def search():
            try:
                similar_solutions = self.agent.find_similar_solutions_with_preview(query, threshold=0.30)
                self.root.after(0, lambda: self.update_kb_results_with_preview(similar_solutions))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=search, daemon=True).start()

    def update_kb_results_with_preview(self, similar_solutions):
        """Update KB results display with previews."""
        self.results_display.config(state=tk.NORMAL)
        self.results_display.delete(1.0, tk.END)
        
        # Clear solution display
        self.solution_display.config(state=tk.NORMAL)
        self.solution_display.delete(1.0, tk.END)
        self.solution_display.config(state=tk.DISABLED)
        
        # Clear stored results data
        self.results_data = []
        
        for i, result in enumerate(similar_solutions):
            filename = result['filename']
            score = result['score']
            preview = result['preview']
            
            # Store result data for click handling
            start_line = float(self.results_display.index(tk.INSERT))
            
            # Add filename and score as clickable header
            self.results_display.insert(tk.END, f"📄 {filename}\n", f"title_{i}")
            self.results_display.insert(tk.END, f"   Relevance: {score:.2f}\n", f"score_{i}")
            
            # Process and add preview with better formatting
            if "**Cause:**" in preview or "**Resolution:**" in preview:
                # Handle structured preview with Cause and Resolution
                self.results_display.insert(tk.END, "   ", "indent")
                
                # Split preview into parts and format each
                parts = preview.split('\n\n')
                for j, part in enumerate(parts):
                    if part.startswith("**Cause:**"):
                        cause_text = part.replace("**Cause:**", "").strip()
                        self.results_display.insert(tk.END, "Cause: ", f"label_{i}")
                        self.results_display.insert(tk.END, f"{cause_text}\n", f"content_{i}")
                    elif part.startswith("**Resolution:**"):
                        resolution_text = part.replace("**Resolution:**", "").strip()
                        self.results_display.insert(tk.END, "   Resolution: ", f"label_{i}")
                        self.results_display.insert(tk.END, f"{resolution_text}\n", f"content_{i}")
                    
                    if j < len(parts) - 1:  # Add spacing between parts
                        self.results_display.insert(tk.END, "   \n", "spacing")
            else:
                # Handle regular preview
                self.results_display.insert(tk.END, f"   Preview: {preview}\n", f"preview_{i}")
            
            self.results_display.insert(tk.END, "\n" + "─"*80 + "\n\n", "separator")
            
            end_line = float(self.results_display.index(tk.INSERT))
            
            # Store result data with line ranges
            self.results_data.append({
                'filename': filename,
                'start_line': start_line,
                'end_line': end_line
            })
            
            # Configure tags for styling
            self.results_display.tag_config(f"title_{i}", 
                                          font=('Segoe UI', 11, 'bold'),
                                          foreground=self.colors['primary'])
            self.results_display.tag_config(f"score_{i}",
                                          font=('Segoe UI', 9),
                                          foreground=self.colors['secondary'])
            self.results_display.tag_config(f"label_{i}",
                                          font=('Segoe UI', 9, 'bold'),
                                          foreground=self.colors['success'])
            self.results_display.tag_config(f"content_{i}",
                                          font=('Segoe UI', 9),
                                          foreground=self.colors['text'],
                                          lmargin1=30, lmargin2=30)
            self.results_display.tag_config(f"preview_{i}",
                                          font=('Segoe UI', 9),
                                          foreground=self.colors['text'],
                                          lmargin1=20, lmargin2=20)
            self.results_display.tag_config("separator",
                                          foreground='#CCCCCC')
            self.results_display.tag_config("indent",
                                          lmargin1=20)
        
        if not similar_solutions:
            self.results_display.insert(tk.END, "No results found. Try different search terms.")
        
        self.results_display.config(state=tk.DISABLED)

    def on_result_click(self, event):
        """Handle clicks on search results."""
        # Get the line number where the click occurred
        click_line = float(self.results_display.index(f"@{event.x},{event.y}"))
        
        # Find which result was clicked
        for result in self.results_data:
            if result['start_line'] <= click_line <= result['end_line']:
                # Show the full solution for this result
                filename = result['filename']
                solution = self.agent.get_solution(filename)
                
                self.solution_display.config(state=tk.NORMAL)
                self.solution_display.delete(1.0, tk.END)
                self.solution_display.insert(tk.END, solution)
                self.solution_display.config(state=tk.DISABLED)
                break

def main():
    root = tk.Tk()
    app = TechSupportGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 