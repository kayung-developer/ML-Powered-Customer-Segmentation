import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

class MLClusterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML-Powered Customer Segmentation")
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")

        # Variables
        self.data = None
        self.cluster_algorithm = ctk.StringVar(value="K-Means")
        self.num_clusters = ctk.IntVar(value=3)
        self.eps_value = ctk.DoubleVar(value=0.5)
        self.min_samples = ctk.IntVar(value=5)
        self.selected_features = []

        # UI Components
        self.create_widgets()

    def create_widgets(self):
        # Layout
        self.left_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_frame.pack(side="left", fill="y")

        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.pack(side="right", expand=True, fill="both")

        # Buttons & Menus
        ctk.CTkLabel(self.left_frame, text="Data Upload", font=("Arial", 16, "bold")).pack(pady=10)
        self.upload_button = ctk.CTkButton(self.left_frame, text="Upload CSV", command=self.upload_data)
        self.upload_button.pack(pady=10)

        ctk.CTkLabel(self.left_frame, text="Clustering Options", font=("Arial", 16, "bold")).pack(pady=10)

        # Algorithm Selection
        ctk.CTkLabel(self.left_frame, text="Choose Algorithm:", font=("Arial", 14)).pack(pady=5)
        algo_menu = ctk.CTkOptionMenu(self.left_frame, values=["K-Means", "DBSCAN"], variable=self.cluster_algorithm, command=self.toggle_parameters)
        algo_menu.pack(pady=5)

        # K-Means Parameters
        self.kmeans_frame = ctk.CTkFrame(self.left_frame)
        ctk.CTkLabel(self.kmeans_frame, text="Number of Clusters:", font=("Arial", 14)).pack(pady=5)
        ctk.CTkEntry(self.kmeans_frame, textvariable=self.num_clusters, width=150).pack(pady=5)
        self.kmeans_frame.pack(pady=10)

        # DBSCAN Parameters
        self.dbscan_frame = ctk.CTkFrame(self.left_frame)
        ctk.CTkLabel(self.dbscan_frame, text="Epsilon (eps):", font=("Arial", 14)).pack(pady=5)
        ctk.CTkEntry(self.dbscan_frame, textvariable=self.eps_value, width=150).pack(pady=5)
        ctk.CTkLabel(self.dbscan_frame, text="Min Samples:", font=("Arial", 14)).pack(pady=5)
        ctk.CTkEntry(self.dbscan_frame, textvariable=self.min_samples, width=150).pack(pady=5)

        # Run Clustering Button
        self.run_button = ctk.CTkButton(self.left_frame, text="Run Clustering", bg_color="green", width=90, command=self.run_clustering)
        self.run_button.pack(pady=20)

        # Results & Output Text
        self.output_text = ctk.CTkTextbox(self.left_frame, height=300, width=150)
        self.output_text.pack(pady=20)
        self.output_text.insert("0.0", "Output will be shown here...")

        self.plot_frame = ctk.CTkFrame(self.right_frame)
        self.plot_frame.pack(fill="both", expand=True)

    def toggle_parameters(self, algo):
        if algo == "K-Means":
            self.kmeans_frame.pack(pady=10)
            self.dbscan_frame.pack_forget()
        elif algo == "DBSCAN":
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack(pady=10)

    def upload_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", f"Data loaded successfully with shape {self.data.shape}.")
                self.output_text.insert("0.0", f"Data loaded successfully with shape {self.data.shape}.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
                self.output_text.insert("0.0", f"Failed to load data: {e}")

    def run_clustering(self):
        if self.data is None:
            messagebox.showerror("Error", "Please upload a dataset first.")
            self.output_text.insert("0.0", "Please upload a dataset first.")
            return

        try:
            numeric_data = self.data.select_dtypes(include=np.number)
            scaled_data = StandardScaler().fit_transform(numeric_data)
            algo = self.cluster_algorithm.get()

            if algo == "K-Means":
                n_clusters = self.num_clusters.get()
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(scaled_data)
                centroids = model.cluster_centers_
                silhouette = silhouette_score(scaled_data, labels)
                self.output_text.insert("0.0", f"Silhouette Score: {silhouette:.2f}")
                self.plot_clusters(numeric_data, labels, centroids)

            elif algo == "DBSCAN":
                eps = self.eps_value.get()
                min_samples = self.min_samples.get()
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(scaled_data)
                self.output_text.insert("0.0", f"DBSCAN Clustering Complete.")
                self.plot_clusters(numeric_data, labels)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform clustering: {e}")
            self.output_text.insert("0.0", f"Failed to perform clustering: {e}")

    def plot_clusters(self, data, labels, centroids=None):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis", ax=ax, s=100)
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
        ax.set_title("Cluster Visualization", fontsize=16)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

if __name__ == "__main__":
    app = MLClusterApp()
    app.mainloop()
