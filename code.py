### Import Libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings

### Load Data

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return None

# Filepath to the dataset
filepath = '2025-VeloCityX-Expanded-Fan-Engagement-Data.csv'

# Load the data
velocity_df = load_data(filepath)

### Data Inspection

def inspect_data(df):
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print(f"\nShape of the dataset: {df.shape}")
    print(f"Size of the dataset: {df.size}")

    print("\nStatistical summary of the dataset:")
    print(df.describe())

    print("\nChecking for missing values:")
    print(df.isnull().sum())

inspect_data(velocity_df)

### Data Cleaning

def handle_missing_values(df):
    df_cleaned = df.copy()
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].median())
    print("\nMissing values after imputation:")
    print(df_cleaned.isnull().sum())
    return df_cleaned

def convert_data_types(df):
    df_converted = df.copy()
    if 'User ID' in df_converted.columns:
        df_converted['User ID'] = df_converted['User ID'].astype(str)
    return df_converted

# Handle missing values
velocity_df_cleaned = handle_missing_values(velocity_df)

# Convert data types
velocity_df_cleaned = convert_data_types(velocity_df_cleaned)

### Feature Selection and Correlation Analysis

def plot_correlation_matrix(df, title='Correlation Matrix', figsize=(12,10)):
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    print(f"\nRemoved {len(to_drop)} highly correlated features: {to_drop}")
    return df_reduced

# Feature selection: exclude 'User ID' for correlation and clustering
if 'User ID' in velocity_df_cleaned.columns:
    features_df = velocity_df_cleaned.drop(columns=['User ID'])
else:
    features_df = velocity_df_cleaned.copy()

# Plot correlation matrix
plot_correlation_matrix(features_df, title='Correlation Matrix for VeloCityX Data')

# Remove highly correlated features
features_reduced = remove_highly_correlated_features(features_df)

### Outlier Detection and Removal

def detect_and_remove_outliers(df, features, threshold=3):
    df_out = df.copy()
    z_scores = np.abs(stats.zscore(df_out[features]))
    filtered_entries = (z_scores < threshold).all(axis=1)
    outliers_removed = df_out[filtered_entries]
    print(f"\nRemoved {df_out.shape[0] - outliers_removed.shape[0]} outliers.")
    return outliers_removed

# Outlier detection and removal
numerical_features = features_reduced.select_dtypes(include=[np.number]).columns.tolist()
velocity_df_cleaned = detect_and_remove_outliers(velocity_df_cleaned, numerical_features)

### Clustering Analysis

def plot_scatter(x, y, data, title, xlabel, ylabel, alpha=0.6):

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, data=data, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def perform_clustering(df, features, n_clusters=3):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    cluster_labels = pipeline.fit_predict(df[features])
    return cluster_labels, pipeline

def determine_optimal_clusters(df, features, max_k=10):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    inertia_list = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia_list.append(kmeans.inertia_)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)

    # Plot Elbow Method
    plt.figure(figsize=(10,5))
    plt.plot(range(2, max_k + 1), inertia_list, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(10,5))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Determine optimal k as the one with the highest Silhouette Score
    optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")
    return optimal_k

# Determine optimal number of clusters
selected_features = features_reduced.columns.tolist()
optimal_k = determine_optimal_clusters(velocity_df_cleaned, selected_features, max_k=10)

# Perform clustering with the optimal number of clusters
cluster_labels, clustering_pipeline = perform_clustering(velocity_df_cleaned, selected_features, n_clusters=optimal_k)
velocity_df_cleaned['Cluster'] = cluster_labels

### Principal Component Analysis (PCA)

def perform_pca(df, features, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(df[features])
    print(f"\nExplained variance by {n_components} components: {pca.explained_variance_ratio_}")
    return components, pca

def plot_pca_clusters(components, labels, title='PCA Clusters', figsize=(8,6)):
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    plt.figure(figsize=figsize)
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='Set1', alpha=0.7)
    plt.title(title)
    plt.show()

def plot_pca_clusters_detailed(components, labels, title='PCA Clusters Detailed', figsize=(10,8)):
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    plt.figure(figsize=figsize)
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='Set1', alpha=0.7, s=100, edgecolor='k')
    plt.title(title)
    plt.legend(title='Cluster')
    plt.show()

def plot_pca_feature_importance(pca_model, features):
    components = pca_model.components_
    
    plt.figure(figsize=(12,6))
    
    for i, component in enumerate(components):
        plt.bar(features, component, alpha=0.6, label=f'PC{i+1}')
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.title('Feature Importance in PCA Components', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Component Weights', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Perform PCA for visualization
pca_components, pca_model = perform_pca(velocity_df_cleaned, selected_features, n_components=2)

# Plot PCA clusters
plot_pca_clusters(pca_components, velocity_df_cleaned['Cluster'], title='User Clusters Visualization based on PCA')

# Plot detailed PCA clusters
plot_pca_clusters_detailed(pca_components, velocity_df_cleaned['Cluster'], title='User Clusters Visualization based on PCA (Detailed)')

# Plot PCA feature importance
plot_pca_feature_importance(pca_model, selected_features)

### Merchandise Purchase Trends

def merchandise_purchase_trends(df):

    # Users with at least one purchase
    purchasers = df[df['Virtual Merchandise Purchases'] > 0]
    non_purchasers = df[df['Virtual Merchandise Purchases'] == 0]

    print(f"\nTotal Purchasers: {purchasers.shape[0]}")
    print(f"Total Non-Purchasers: {non_purchasers.shape[0]}")

    # Select only numeric columns to avoid errors
    numeric_df = df.select_dtypes(include=[np.number])

    # Ensure 'Virtual Merchandise Purchases' is included in the groupby
    if 'Virtual Merchandise Purchases' not in numeric_df.columns:
        print("Error: 'Virtual Merchandise Purchases' column is missing or not numeric.")
        return

    # Compare mean values of features between purchasers and non-purchasers
    comparison = numeric_df.groupby('Virtual Merchandise Purchases').mean()
    print("\nComparison of Feature Means Between Purchasers and Non-Purchasers:")
    print(comparison)

    # Visualize differences
    features = ['Fan Challenges Completed', 'Predictive Accuracy (%)', 
                'Sponsorship Interactions (Ad Clicks)', 
                'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']
    
    for feature in features:
        plt.figure(figsize=(8,6))
        sns.kdeplot(purchasers[feature], fill=True, label='Purchasers')
        sns.kdeplot(non_purchasers[feature], fill=True, label='Non-Purchasers')
        plt.title(f'Distribution of {feature} by Purchase Status')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

merchandise_purchase_trends(velocity_df_cleaned)

### Predictive Modeling

def predictive_modeling(df, target='Virtual Merchandise Purchases'):
    # Define features and target
    X = df.drop(columns=['User ID', target, 'Cluster'])  # Exclude 'User ID' and 'Cluster' from features
    y = df[target]

    # Convert target to binary: 1 if purchases > 0, else 0
    y_binary = (y > 0).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"{name} ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature Importance for Random Forest
    rf_model = models['Random Forest']
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importances from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

predictive_modeling(velocity_df_cleaned, target='Virtual Merchandise Purchases')

### Confusion Matrix and ROC Curve for Random Forest

def evaluate_random_forest(df, target='Virtual Merchandise Purchases'):
    # Define features and target
    X = df.drop(columns=['User ID', target, 'Cluster'])  # Exclude 'User ID' and 'Cluster' from features
    y = df[target]

    # Convert target to binary: 1 if purchases > 0, else 0
    y_binary = (y > 0).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve for Random Forest')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Evaluate Random Forest with additional metrics
evaluate_random_forest(velocity_df_cleaned, target='Virtual Merchandise Purchases')
