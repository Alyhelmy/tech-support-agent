import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tech_support_agent import TechSupportAgent
import threading

class TechSupportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Support Agent")
        self.root.geometry("1000x700")  # Increased window size
        
        # Set theme colors
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#1976D2',
            'background': '#F5F5F5',
            'text': '#333333',
            'success': '#4CAF50',
            'warning': '#FFC107'
        }
        
        # Initialize the tech support agent
        self.agent = TechSupportAgent()
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam theme as base
        
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
        self.style.configure("Search.TButton",
                           font=('Segoe UI', 10, 'bold'),
                           padding=10)
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
        self.main_frame.rowconfigure(2, weight=1)
        
        # Create and place widgets
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = ttk.Label(
            self.main_frame,
            text="Tech Support Assistant",
            style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Query Frame
        query_frame = ttk.Frame(self.main_frame)
        query_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        query_frame.columnconfigure(1, weight=1)

        # Query Label
        query_label = ttk.Label(
            query_frame,
            text="Describe your issue:",
            font=('Segoe UI', 11)
        )
        query_label.grid(row=0, column=0, padx=(0, 10))

        # Query Entry
        self.query_entry = ttk.Entry(
            query_frame,
            width=50,
            font=('Segoe UI', 11)
        )
        self.query_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.query_entry.bind("<Return>", lambda e: self.process_query())

        # Search Button
        search_button = ttk.Button(
            query_frame,
            text="Search",
            command=self.process_query,
            style="Search.TButton"
        )
        search_button.grid(row=0, column=2)

        # Results Frame
        results_frame = ttk.LabelFrame(
            self.main_frame,
            text="Search Results",
            padding="10"
        )
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results Text
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=('Segoe UI', 11),
            background='white',
            foreground=self.colors['text']
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_text.config(state=tk.DISABLED)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(10, 5)
        )
        status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Bottom Buttons Frame
        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.grid(row=4, column=0, pady=(0, 10))
        
        # Database Info Button
        db_info_button = ttk.Button(
            bottom_frame,
            text="Show Database Info",
            command=self.show_database_info,
            style="TButton"
        )
        db_info_button.grid(row=0, column=0, padx=5)
        
        # Refresh Knowledge Base Button
        refresh_button = ttk.Button(
            bottom_frame,
            text="Refresh Knowledge Base",
            command=self.refresh_knowledge_base,
            style="TButton"
        )
        refresh_button.grid(row=0, column=1, padx=5)

    def update_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def process_query(self):
        query = self.query_entry.get().strip()
        if not query:
            return

        self.status_var.set("Processing query...")
        self.update_results("Searching for solutions...\n")

        # Process query in a separate thread to keep GUI responsive
        def process():
            similar_solutions = self.agent.find_similar_solutions(query)
            
            if not similar_solutions:
                self.root.after(0, lambda: self.update_results(
                    "No relevant solutions found. Please try rephrasing your query."
                ))
                self.root.after(0, lambda: self.status_var.set("No results found"))
                return

            # Create a selection window for multiple results
            if len(similar_solutions) > 1:
                self.create_selection_window(similar_solutions)
            else:
                solution = self.agent.get_solution(similar_solutions[0][0])
                self.root.after(0, lambda: self.update_results(solution))
                self.root.after(0, lambda: self.status_var.set("Solution found"))

        threading.Thread(target=process, daemon=True).start()

    def create_selection_window(self, similar_solutions):
        # Create a new top-level window
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Topic")
        selection_window.geometry("800x500")
        
        # Create a frame for the listbox
        frame = ttk.Frame(selection_window, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        selection_window.columnconfigure(0, weight=1)
        selection_window.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Create a label
        label = ttk.Label(
            frame,
            text="Multiple solutions found. Please select the most relevant topic:",
            font=('Segoe UI', 11),
            wraplength=700
        )
        label.grid(row=0, column=0, pady=(0, 15))
        
        # Create a listbox with scrollbar
        listbox_frame = ttk.Frame(frame)
        listbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        listbox = tk.Listbox(
            listbox_frame,
            width=80,
            height=15,
            font=('Segoe UI', 11),
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set
        )
        listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=listbox.yview)
        
        # Add items to the listbox
        for filename, score in similar_solutions:
            listbox.insert(tk.END, f"{filename} (Relevance: {score:.2f})")
        
        # Create a button frame
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, pady=(20, 0))
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                filename = similar_solutions[index][0]
                solution = self.agent.get_solution(filename)
                self.update_results(solution)
                self.status_var.set("Solution displayed")
                selection_window.destroy()
        
        # Create Select button
        select_button = ttk.Button(
            button_frame,
            text="Select",
            command=on_select,
            style="Search.TButton"
        )
        select_button.grid(row=0, column=0, padx=5)
        
        # Create Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=selection_window.destroy,
            style="TButton"
        )
        cancel_button.grid(row=0, column=1, padx=5)

    def show_database_info(self):
        """Show information about the stored knowledge base."""
        knowledge_count = self.agent.knowledge_store.get_knowledge_count()
        if knowledge_count > 0:
            message = f"Knowledge Base Status:\n\n"
            message += f"Total entries: {knowledge_count}\n"
            message += f"Database file: {self.agent.knowledge_store.db_path}\n\n"
            message += "The system has learned from your knowledge base files and stored the information in the database. You can now safely delete the original text files if desired."
        else:
            message = "No knowledge base entries found in the database."
        
        messagebox.showinfo("Database Information", message)

    def refresh_knowledge_base(self):
        """Refresh the knowledge base with any new files."""
        self.status_var.set("Refreshing knowledge base...")
        self.update_results("Refreshing knowledge base...\n")
        
        def refresh():
            try:
                self.agent.load_knowledge_base()
                self.root.after(0, lambda: self.status_var.set("Knowledge base refreshed"))
                self.root.after(0, lambda: self.update_results("Knowledge base has been refreshed with any new files."))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("Error refreshing knowledge base"))
                self.root.after(0, lambda: self.update_results(f"Error refreshing knowledge base: {str(e)}"))
        
        threading.Thread(target=refresh, daemon=True).start()

def main():
    root = tk.Tk()
    app = TechSupportGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 