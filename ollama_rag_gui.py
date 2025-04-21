#!/usr/bin/env python3
"""
Ollama RAG GUI Application
--------------------------
A graphical user interface for the Ollama RAG system.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import traceback
import logging
import json

# Import the Ollama RAG system
from ollama_rag import OllamaRAG, logger, OLLAMA_API_BASE
import requests

class RedirectText:
    """Redirect print statements to a tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_timer = None
    
    def write(self, string):
        self.queue.put(string)
        if self.update_timer is None:
            self.update_timer = self.text_widget.after(100, self.update_text)
    
    def update_text(self):
        while not self.queue.empty():
            text = self.queue.get()
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")
        self.update_timer = None
    
    def flush(self):
        pass

class LogHandler(logging.Handler):
    """Custom logging handler for redirecting logs to a tkinter Text widget."""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_timer = None
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)
    
    def emit(self, record):
        msg = self.format(record)
        self.queue.put(msg + "\n")
        if self.update_timer is None:
            self.update_timer = self.text_widget.after(100, self.update_log)
    
    def update_log(self):
        while not self.queue.empty():
            msg = self.queue.get()
            self.text_widget.configure(state="normal")
            
            # Apply color based on log level
            if "ERROR" in msg:
                self.text_widget.insert(tk.END, msg, "error")
            elif "WARNING" in msg:
                self.text_widget.insert(tk.END, msg, "warning")
            elif "INFO" in msg:
                self.text_widget.insert(tk.END, msg, "info")
            else:
                self.text_widget.insert(tk.END, msg)
                
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")
        self.update_timer = None

class OllamaRagGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure the window
        self.title("Ollama RAG GUI")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Create a style for ttk widgets
        self.style = ttk.Style()
        self.style.configure("TFrame", padding=5)
        self.style.configure("TLabelframe", padding=10)
        self.style.configure("TButton", padding=5)
        
        # Create the main notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.index_tab = ttk.Frame(self.notebook)
        self.query_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.index_tab, text="Index Documents")
        self.notebook.add(self.query_tab, text="Query Knowledge Base")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Create the logging text widget
        self.logs_text = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD, height=20)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.logs_text.configure(state="disabled")
        
        # Set up log tag styling
        self.logs_text.tag_configure("error", foreground="red")
        self.logs_text.tag_configure("warning", foreground="orange")
        self.logs_text.tag_configure("info", foreground="blue")
        
        # Set up logging redirection
        self.log_handler = LogHandler(self.logs_text)
        logger.addHandler(self.log_handler)
        
        # Initialize RAG object
        self.rag = None
        
        # Available models list
        self.available_models = []
        
        # Get available models in a separate thread to avoid blocking UI
        threading.Thread(target=self.fetch_available_models, daemon=True).start()
        
        # Create UI elements for each tab
        self.create_index_tab()
        self.create_query_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def set_status(self, message):
        """Thread-safe way to update the status bar."""
        self.after(0, lambda: self.status_var.set(message))
    
    def fetch_available_models(self):
        """Fetch available models from Ollama."""
        try:
            self.set_status("Fetching available models...")
            response = requests.get(f"{OLLAMA_API_BASE}/tags")
            
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                self.available_models = [model["name"] for model in models_data]
                
                if self.available_models:
                    # Update model comboboxes in both tabs
                    self.update_model_comboboxes()
                    self.set_status(f"Found {len(self.available_models)} models")
                    logger.info(f"Found {len(self.available_models)} models: {', '.join(self.available_models)}")
                else:
                    self.set_status("No models found in Ollama")
                    logger.warning("No models found in Ollama")
            else:
                self.set_status("Failed to fetch models")
                logger.error(f"Failed to fetch models: {response.text}")
        except Exception as e:
            self.set_status("Error fetching models")
            logger.error(f"Error fetching models: {str(e)}")
    
    def update_model_comboboxes(self):
        """Update model comboboxes with fetched models."""
        if not self.available_models:
            return
            
        # This needs to be called in the main thread
        self.after(0, self._update_comboboxes)
    
    def _update_comboboxes(self):
        """Update comboboxes in the main thread."""
        # Update embedding model combobox
        if hasattr(self, 'embedding_model_combobox'):
            self.embedding_model_combobox['values'] = self.available_models
            # Set a default value if available
            for model in self.available_models:
                if 'embed' in model.lower() or 'minilm' in model.lower():
                    self.embedding_model_var.set(model)
                    break
            if not self.embedding_model_var.get() and self.available_models:
                self.embedding_model_var.set(self.available_models[0])
        
        # Update query model combobox
        if hasattr(self, 'query_model_combobox'):
            self.query_model_combobox['values'] = self.available_models
            # Set a sensible default
            for model in self.available_models:
                if any(name in model.lower() for name in ['llama', 'mistral', 'gemma']):
                    self.query_model_var.set(model)
                    break
            if not self.query_model_var.get() and self.available_models:
                self.query_model_var.set(self.available_models[0])
    
    def refresh_models(self):
        """Refresh the list of available models."""
        threading.Thread(target=self.fetch_available_models, daemon=True).start()
    
    def create_index_tab(self):
        """Create UI elements for the indexing tab."""
        # Frame for input directory
        input_frame = ttk.LabelFrame(self.index_tab, text="Input Documents")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Input Directory or Files:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_entry = ttk.Entry(input_frame, width=50)
        self.input_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Create a button frame for the two browse buttons
        browse_btn_frame = ttk.Frame(input_frame)
        browse_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(browse_btn_frame, text="Browse Directory...", command=self.browse_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(browse_btn_frame, text="Browse Files...", command=self.browse_input_files).pack(side=tk.LEFT, padx=2)
        
        # Information label about supported file types
        ttk.Label(input_frame, text="Supported file types: Markdown (.md), Text (.txt)", 
                  font=("", 8), foreground="gray").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5)
        
        # Frame for output file
        output_frame = ttk.LabelFrame(self.index_tab, text="Output")
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output JSON File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(output_frame, width=50)
        self.output_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(output_frame, text="Save As...", command=self.browse_output).grid(row=0, column=2, padx=5, pady=5)
        
        # Frame for model selection and options
        model_frame = ttk.LabelFrame(self.index_tab, text="Model & Options")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Embedding Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.embedding_model_var = tk.StringVar(value="all-minilm")
        
        # Create a combobox instead of an entry
        self.embedding_model_combobox = ttk.Combobox(model_frame, textvariable=self.embedding_model_var, width=30)
        self.embedding_model_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add a refresh button for models
        ttk.Button(model_frame, text="Refresh Models", command=self.refresh_models).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Chunk Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.chunk_size_var = tk.IntVar(value=1000)
        self.chunk_size_entry = ttk.Entry(model_frame, textvariable=self.chunk_size_var, width=10)
        self.chunk_size_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Index button
        index_button_frame = ttk.Frame(self.index_tab)
        index_button_frame.pack(fill=tk.X, pady=10)
        
        self.index_button = ttk.Button(index_button_frame, text="Index Documents", command=self.index_documents)
        self.index_button.pack(side=tk.RIGHT, padx=5)
    
    def create_query_tab(self):
        """Create UI elements for the query tab."""
        # Frame for index file
        index_frame = ttk.LabelFrame(self.query_tab, text="Knowledge Base")
        index_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(index_frame, text="Index File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.index_entry = ttk.Entry(index_frame, width=50)
        self.index_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(index_frame, text="Browse...", command=self.browse_index).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(index_frame, text="Load", command=self.load_index).grid(row=0, column=3, padx=5, pady=5)
        
        # Frame for model selection
        model_query_frame = ttk.LabelFrame(self.query_tab, text="Model")
        model_query_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_query_frame, text="Generation Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.query_model_var = tk.StringVar(value="llama3")
        
        # Create a combobox instead of an entry
        self.query_model_combobox = ttk.Combobox(model_query_frame, textvariable=self.query_model_var, width=30)
        self.query_model_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add a refresh button for models
        ttk.Button(model_query_frame, text="Refresh Models", command=self.refresh_models).grid(row=0, column=2, padx=5, pady=5)
        
        # Chat frame
        chat_frame = ttk.LabelFrame(self.query_tab, text="Chat")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=15)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.configure(state="disabled")
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        self.query_entry = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=3)
        self.query_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        self.query_entry.bind("<Control-Return>", self.send_query)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_query)
        self.send_button.pack(fill=tk.X, pady=(0, 5))
        
        self.clear_button = ttk.Button(button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(fill=tk.X)
    
    def browse_input(self):
        """Open dialog to select input directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, directory)
    
    def browse_input_files(self):
        """Open dialog to select input files."""
        files = filedialog.askopenfilenames(
            filetypes=[("Document files", "*.md *.txt"), ("Markdown files", "*.md"), 
                      ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if files:
            # Convert tuple to semicolon-separated string for display
            files_str = ";".join(files)
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, files_str)
    
    def browse_output(self):
        """Open dialog to select output file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
    
    def browse_index(self):
        """Open dialog to select index file."""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, filename)
    
    def index_documents(self):
        """Start indexing in a separate thread."""
        input_dir = self.input_entry.get().strip()
        output_file = self.output_entry.get().strip()
        embedding_model = self.embedding_model_var.get().strip()
        chunk_size = self.chunk_size_var.get()
        
        if not input_dir or not output_file:
            messagebox.showerror("Error", "Input directory/files and output file are required")
            return
        
        def do_indexing():
            self.set_status("Indexing documents...")
            self.after(0, lambda: self.index_button.configure(state="disabled"))
            
            try:
                # Initialize RAG
                self.rag = OllamaRAG(embedding_model=embedding_model)
                
                # Check if input contains multiple files (semicolon-separated)
                if ";" in input_dir:
                    # Process each file individually
                    file_paths = input_dir.split(";")
                    total_docs = 0
                    
                    for file_path in file_paths:
                        file_path = file_path.strip()
                        if not file_path:
                            continue
                            
                        # Process this file 
                        self.set_status(f"Processing file: {os.path.basename(file_path)}...")
                        self.rag.load_document_files(file_path, max_chunk_size=chunk_size)
                        total_docs += 1
                        
                    self.set_status(f"Processed {total_docs} files.")
                else:
                    # Process directory or single file pattern
                    self.rag.load_document_files(input_dir, max_chunk_size=chunk_size)
                
                # Create embeddings for all loaded documents
                self.rag.create_embeddings()
                self.rag.save_to_json(output_file)
                
                self.set_status(f"Indexing completed. Output saved to {output_file}")
                self.after(0, lambda: messagebox.showinfo("Success", f"Indexing completed. Output saved to {output_file}"))
                
                # Update the index entry in the query tab
                def update_index_entry():
                    self.index_entry.delete(0, tk.END)
                    self.index_entry.insert(0, output_file)
                self.after(0, update_index_entry)
                
            except Exception as e:
                error_msg = f"Error during indexing: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                self.set_status("Error during indexing")
                self.after(0, lambda: messagebox.showerror("Error", f"Error during indexing: {str(e)}"))
            finally:
                self.after(0, lambda: self.index_button.configure(state="normal"))
        
        # Start indexing in a separate thread
        threading.Thread(target=do_indexing, daemon=True).start()
    
    def load_index(self):
        """Load the index file."""
        index_file = self.index_entry.get().strip()
        if not index_file:
            messagebox.showerror("Error", "Please select an index file")
            return
        
        if not os.path.exists(index_file):
            messagebox.showerror("Error", f"Index file not found: {index_file}")
            return
        
        try:
            self.set_status("Loading index...")
            
            # Initialize the RAG object with the user's chosen model
            generation_model = self.query_model_var.get().strip()
            self.rag = OllamaRAG(llm_model=generation_model)
            
            # Load the index
            self.rag.load_from_json(index_file)
            
            self.set_status(f"Loaded index with {len(self.rag.documents)} documents")
            self.update_chat_display(f"System: Loaded index with {len(self.rag.documents)} documents.")
        except Exception as e:
            error_msg = f"Error loading index: {str(e)}"
            logger.error(error_msg)
            self.set_status("Error loading index")
            messagebox.showerror("Error", error_msg)
    
    def send_query(self, event=None):
        """Send a query to the RAG system."""
        query = self.query_entry.get("1.0", tk.END).strip()
        if not query:
            return
        
        if not self.rag or not self.rag.documents:
            messagebox.showerror("Error", "Please load an index first")
            return
        
        # Update chat with user's query
        self.update_chat_display(f"You: {query}")
        
        # Clear the input field
        self.query_entry.delete("1.0", tk.END)
        
        def do_query():
            try:
                self.set_status("Generating response...")
                self.after(0, lambda: self.send_button.configure(state="disabled"))
                
                # Add a thinking indicator
                def update_thinking():
                    self.update_chat_display("Ollama: Thinking...")
                self.after(0, update_thinking)
                
                # Update the model in case it changed
                self.rag.llm_model = self.query_model_var.get().strip()
                
                # Generate response
                response = self.rag.generate_response(query)
                
                # Remove the thinking indicator and add the real response
                def update_response():
                    self.remove_last_message()
                    self.update_chat_display(f"Ollama: {response}")
                self.after(0, update_response)
                
                self.set_status("Ready")
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                self.set_status("Error generating response")
                
                # Remove the thinking indicator and add the error message
                def update_error():
                    self.remove_last_message()
                    self.update_chat_display(f"Error: {str(e)}")
                self.after(0, update_error)
            finally:
                self.after(0, lambda: self.send_button.configure(state="normal"))
        
        # Start query in a separate thread
        threading.Thread(target=do_query, daemon=True).start()
    
    def update_chat_display(self, message):
        """Add a message to the chat display."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")
    
    def remove_last_message(self):
        """Remove the last message from the chat display."""
        self.chat_display.configure(state="normal")
        
        # Get the text content
        content = self.chat_display.get("1.0", tk.END)
        
        # Find the last two newlines
        lines = content.split("\n")
        if len(lines) > 3:  # Need at least 4 lines to remove the last message
            # Keep all but the last message and the empty lines after it
            new_content = "\n".join(lines[:-3])
            
            # Clear and re-insert
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.insert("1.0", new_content + "\n")
        
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state="disabled")


if __name__ == "__main__":
    app = OllamaRagGUI()
    app.mainloop() 