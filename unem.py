
# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
file_path = r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\unemployment analysis.xlsx"
df = pd.read_excel(file_path)

# Convert all column names to strings
df.columns = df.columns.astype(str)

# Define the navigation menu
nav = st.sidebar.radio("Forecasting the Future: Navigating the Unknowns of Unemployment", [
    "Dashboard", "Dataset", "Summary Statistics", "Missing Data", "Correlation",
    "SVM Model", "Regression", "Chi-square Test", "Factor Analysis", "PCA Visualization",
    "Hierarchical Clustering", "Agglomerative Clustering", "KMeans Clustering"
])

# Dashboard page
if nav == "Dashboard":
    st.markdown("<h1 style='text-align: center; color: #4285F4;'>📊 Dashboard</h1>", unsafe_allow_html=True)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Missing Data
    st.subheader("Missing Data")
    st.write(df.isnull().sum())
    
    # Visualize missing data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, ax=ax)
    st.pyplot(fig)

    # Correlation
    st.subheader("Correlation")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # PCA Visualization
    st.subheader("Principal Component Analysis (PCA)")
    X = df.drop("Country Code", axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Visualize PCA
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=ax)
    ax.set_title('Principal Component Analysis (PCA)')
    st.pyplot(fig)

    # Clustering
    st.subheader("Clustering")
    
    # Hierarchical Clustering
    st.markdown("<h3 style='color: #4285F4;'>🌳 Hierarchical Clustering</h3>", unsafe_allow_html=True)
    merged = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(merged, leaf_rotation=90, leaf_font_size=8, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # Agglomerative Clustering
    st.markdown("<h3 style='color: #0F9D58;'>🔳 Agglomerative Clustering</h3>", unsafe_allow_html=True)
    agglomerative = AgglomerativeClustering(n_clusters=3)
    clusters = agglomerative.fit_predict(X_scaled)
    df["Cluster"] = clusters
    st.write("Cluster Assignment:")
    st.write(df["Cluster"].value_counts())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Cluster"], palette="viridis", ax=ax)
    plt.title("Agglomerative Clustering")
    st.pyplot(fig)

    # Evaluate clustering performance using silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    st.write(f"Silhouette Score: {silhouette_avg}")

    # KMeans Clustering
    st.markdown("<h3 style='color: #FF5733;'>📍 KMeans Clustering</h3>", unsafe_allow_html=True)
    max_clusters = 10
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. Number of Clusters')
    st.pyplot(fig)


    # PCA Visualization
    st.subheader("Principal Component Analysis (PCA)")
    X = df.drop("Country Code", axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Visualize PCA
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=ax)
    ax.set_title('Principal Component Analysis (PCA)')
    st.pyplot(fig)

    # Clustering
    st.subheader("Clustering")
    
    # Hierarchical Clustering
    st.markdown("<h3 style='color: #4285F4;'>🌳 Hierarchical Clustering</h3>", unsafe_allow_html=True)
    merged = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(merged, leaf_rotation=90, leaf_font_size=8, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # Agglomerative Clustering
    st.markdown("<h3 style='color: #0F9D58;'>🔳 Agglomerative Clustering</h3>", unsafe_allow_html=True)
    agglomerative = AgglomerativeClustering(n_clusters=3)
    clusters = agglomerative.fit_predict(X_scaled)
    df["Cluster"] = clusters
    st.write("Cluster Assignment:")
    st.write(df["Cluster"].value_counts())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Cluster"], palette="viridis", ax=ax)
    plt.title("Agglomerative Clustering")
    st.pyplot(fig)

    # Evaluate clustering performance using silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    st.write(f"Silhouette Score: {silhouette_avg}")

    # KMeans Clustering
    st.markdown("<h3 style='color: #FF5733;'>📍 KMeans Clustering</h3>", unsafe_allow_html=True)
    max_clusters = 10
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. Number of Clusters')
    st.pyplot(fig)



# Dataset page
elif nav == "Dataset":
    st.markdown("<h1 style='text-align: center; color: #4285F4;'>📊 Dataset</h1>", unsafe_allow_html=True)
    st.image("C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\main.png", caption="Travel Agent", use_column_width=True)
    st.write(df)

# Summary Statistics page
elif nav == "Summary Statistics":
    st.markdown("<h1 style='text-align: center; color: #DB4437;'>📈 Summary Statistics</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\desc.png", caption="Descriptive Statistics", use_column_width=True)
    st.write(df.describe())

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate the number of rows and columns for subplots
    num_plots = len(numeric_cols)
    num_rows = (num_plots + 1) // 2
    num_cols = min(2, num_plots)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6 * num_rows))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Boxplot for each numeric column
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=col, data=df, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    st.pyplot(fig)

# Missing Data page
elif nav == "Missing Data":
    st.markdown("<h1 style='text-align: center; color: #F4B400;'>⚠ Missing Data</h1>", unsafe_allow_html=True)
    st.write(df.isnull().sum())

    # Visualize missing data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, ax=ax)
    ax.set_title("Missing Data Visualization")
    st.pyplot(fig)

# Correlation page
elif nav == "Correlation":
    st.markdown("<h1 style='text-align: center; color: #0F9D58;'>🔗 Correlation</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\corr.jpg", caption="Correlation", use_column_width=True)
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# SVM Model page
elif nav == "SVM Model":
    st.markdown("<h1 style='text-align: center; color: #4285F4;'>🤖 SVM Model</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\svm.jpg", caption="SVM Model", use_column_width=True)
    X = df.drop("Country Code", axis=1)
    y = df["Country Code"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC()
    svm.fit(X_scaled, y)
    st.write("SVM Model Accuracy:", svm.score(X_scaled, y))

# Regression page
elif nav == "Regression":
    st.markdown("<h1 style='text-align: center; color: #DB4437;'>📈 Regression</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\reg.jpg", caption="Regression", use_column_width=True)

    # Prepare data for regression
    years = df.columns[1:].astype(int).values.reshape(-1, 1)
    y = df.iloc[:, 1:].values
    min_samples = min(len(years), len(y))
    years = years[:min_samples]
    y = y[:min_samples]

    # Train the regression model
    regressor = LinearRegression()
    regressor.fit(years, y)

    # Predict for a given year
    year_input = st.number_input("Enter a year to predict country codes:", min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)))
    prediction = regressor.predict([[year_input]])

    st.write(f"Predicted country codes for year {year_input}:", prediction.flatten())

    # Future prediction
    future_year = st.number_input("Enter a future year to predict country codes:", min_value=int(max(years)), step=1)
    future_predictions = regressor.predict([[future_year]])
    st.write(f"Predicted country codes for year {future_year}:", future_predictions.flatten())

# Chi-square Test page
elif nav == "Chi-square Test":
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>📊 Chi-square Test</h1>", unsafe_allow_html=True)

    # Select the columns for the Chi-square test
    st.subheader("Select Columns for Chi-square Test")
    column1 = st.selectbox("Select the first column", df.columns)
    column2 = st.selectbox("Select the second column", df.columns)
    
    # Create the contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])
    
    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Display results
    st.write(f"Chi-square statistic: {chi2}")
    st.write(f"P-value: {p}")
    st.write(f"Degrees of freedom: {dof}")
    st.write("Expected frequencies:")
    st.write(expected)

# Factor Analysis page
elif nav == "Factor Analysis":
    st.markdown("<h1 style='text-align: center; color: #FF6F61;'>📊 Factor Analysis</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\fac.png", caption="Factor Analysis", use_column_width=True)
    # Prepare the data
    X_fa = df.drop("Country Code", axis=1)
    
    # Perform Factor Analysis
    fa = FactorAnalyzer(rotation=None)
    fa.fit(X_fa)

    # Print the factor loadings
    st.write("Factor Loadings:")
    st.write(fa.loadings_)

    # Print the variance explained by each factor
    st.write("Variance Explained by Each Factor:")
    st.write(fa.get_factor_variance())

# PCA Visualization page
elif nav == "PCA Visualization":
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🔍 Principal Component Analysis (PCA)</h1>", unsafe_allow_html=True)
    
    # Prepare the data
    X = df.drop("Country Code", axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    
    # Display the PCA DataFrame
    st.subheader("Principal Component Analysis (PCA) Results")
    st.write(pca_df)
    
    # Visualize PCA
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=ax)
    ax.set_title('Principal Component Analysis (PCA)')
    st.pyplot(fig)

    # Explained variance ratio
    st.write("Explained Variance Ratio:")
    st.write(pca.explained_variance_ratio_)

# Hierarchical Clustering page
elif nav == "Hierarchical Clustering":
    st.markdown("<h1 style='text-align: center; color: #4285F4;'>🌳 Hierarchical Clustering</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\clus.png", caption="Clustering", use_column_width=True)
    # Prepare the data
    X_scaled = StandardScaler().fit_transform(df.drop("Country Code", axis=1))
    
    # Perform hierarchical clustering
    merged = linkage(X_scaled, method="ward")

    # Visualize dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(merged, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

# Agglomerative Clustering page
elif nav == "Agglomerative Clustering":
    st.markdown("<h1 style='text-align: center; color: #0F9D58;'>🔳 Agglomerative Clustering</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\clus.png", caption="Clustering", use_column_width=True)
    # Prepare the data
    X_scaled = StandardScaler().fit_transform(df.drop("Country Code", axis=1))
    
    # Perform Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=3)
    clusters = agglomerative.fit_predict(X_scaled)
    df["Cluster"] = clusters
    
    # Display cluster assignment
    st.write("Cluster Assignment:")
    st.write(df["Cluster"].value_counts())

    # Visualize the clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Cluster"], palette="viridis", ax=ax)
    plt.title("Agglomerative Clustering")
    st.pyplot(fig)

    # Evaluate clustering performance using silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    st.write(f"Silhouette Score: {silhouette_avg}")

# KMeans Clustering page
elif nav == "KMeans Clustering":
    st.markdown("<h1 style='text-align: center; color: #4285F4;'>📍 KMeans Clustering</h1>", unsafe_allow_html=True)
    st.image(r"C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 4\\Predictive analysis\\PA Project\\clus.png", caption="Clustering", use_column_width=True)
    # Prepare the data
    X_scaled = StandardScaler().fit_transform(df.drop("Country Code", axis=1))
    
    max_clusters = 10
    silhouette_scores = []

    # Iterate over different number of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. Number of Clusters')
    st.pyplot(fig)