import tkinter as tk
from tkinter import scrolledtext, ttk, Menu, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi DiagnoBot")
        self.root.geometry("500x500")
        self.root.resizable(width=False, height=False)

        # Menambahkan menu bar
        menubar = Menu(root)
        root.config(menu=menubar)

        # Menu utama
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Menu", menu=file_menu)

        # Submenu "About"
        file_menu.add_command(label="About", command=self.show_about)

        frame = tk.Frame(root, bg='black', bd=5)
        frame.place(relx=0.5, rely=0.5, relwidth=1, relheight=1, anchor='center')

        self.chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, bg='black', font=('Helvetica', 12), fg='white')
        self.chat_display.pack(expand=True, fill=tk.BOTH)

        self.user_input = tk.Entry(frame, font=('Helvetica', 12), relief=tk.FLAT)
        self.user_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=10)

        self.send_button = ttk.Button(frame, text="Kirim", command=self.send_message, style='TButton', width=10)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=10)

        self.chat_display.tag_configure('user', justify='right', foreground='#007BFF', spacing3=8)
        self.chat_display.tag_configure('chatbot', justify='left', foreground='#00ff00', spacing3=8)

        # Pesan pembuka dari chatbot
        opening_message = "Diagno Bot: Halo! Saya Diagno Bot. Tanyakan sesuatu kepada saya. Butuh bantuan? ketik '/help' "
        self.chat_display.insert(tk.END, f"\n{opening_message}\n", 'chatbot')
        self.chat_display.see(tk.END)

    def show_about(self):
        about_message = """
    Aplikasi Chatbot DiagnoBot

    Dibuat oleh:
    Dini Ambarwati
    """
        messagebox.showinfo("About", about_message)

    def send_message(self):
        user_text = self.user_input.get()
        chatbot_response_text = chatbot_response(user_text)
        current_time = datetime.now().strftime("%H:%M")

        # Insert user message with time
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{current_time} You: {user_text}\n", 'user')
        self.chat_display.config(state=tk.DISABLED)

        # Insert ChatBot response with time
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{current_time} Diagno Bot: {chatbot_response_text}\n", 'chatbot')
        self.chat_display.config(state=tk.DISABLED)

        # Scroll to the bottom
        self.chat_display.see(tk.END)

        self.user_input.delete(0, tk.END)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        res = getResponse(ints, intents)
    return res

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotGUI(root)
    root.mainloop()
