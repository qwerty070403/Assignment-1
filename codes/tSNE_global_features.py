#Plotting Image Features t-SNE(Focuses on global features)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from main_nearest_neighbor import train_features, val_features, test_features, train_labels, val_labels, test_labels



# Combine train, validation, and test features to visualize them together
all_features = np.vstack((train_features, val_features, test_features))
all_labels = train_labels + val_labels + test_labels

# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(all_features)

# Create a DataFrame for plotting
import pandas as pd

# Map the labels to numeric values for color coding
label_to_numeric = {label: idx for idx, label in enumerate(set(all_labels))}
numeric_labels = [label_to_numeric[label] for label in all_labels]

df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
df['Label'] = numeric_labels

# Plot the t-SNE result using seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', palette='Set1', data=df, legend='full', s=60, alpha=0.7)

plt.title('t-SNE visualization of Image Features')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
