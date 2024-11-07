import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')




#############################################################################################################################

def train_val_curve_fitting(
    actual_train: np.ndarray,
    predicted_train: np.ndarray,
    actual_validation: np.ndarray,
    predicted_validation: np.ndarray,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plots actual vs predicted values for both training and validation data.

    Parameters:
    - actual_train (np.ndarray): Actual target values for the training set.
    - predicted_train (np.ndarray): Predicted values for the training set.
    - actual_validation (np.ndarray): Actual target values for the validation set.
    - predicted_validation (np.ndarray): Predicted values for the validation set.
    - figsize (tuple): Size of the figure for the plots, default is (12, 5).

    Returns:
    - None: Displays the actual vs. predicted plots for training and validation data.
    """

    # Set up the figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot the actual vs. predicted values for the training set
    ax1.plot(actual_train, color='blue', label='Actual Train')
    ax1.plot(predicted_train, color='red', linestyle=':', label='Predicted Train')
    ax1.set_title('Train Data')
    ax1.legend()
    ax1.minorticks_on()
    ax1.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels for clarity

    # Plot the actual vs. predicted values for the validation set
    ax2.plot(actual_validation, color='blue', label='Actual Validation')
    ax2.plot(predicted_validation, color='red', linestyle=':', label='Predicted Validation')
    ax2.set_title('Validation Data')
    ax2.legend()
    ax2.minorticks_on()
    ax2.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels for clarity

    # Find the global min and max across both train and validation sets to ensure consistent y-limits
    ymin = float(min(min(actual_train), min(actual_validation), min(predicted_train), min(predicted_validation)))
    ymax = float(max(max(actual_train), max(actual_validation), max(predicted_train), max(predicted_validation)))

    # Adjust ymin and ymax with scaling for better visual clarity
    if ymin < 0:
        ymin *= 1.1
    elif ymin == 0:
        ymin -= 0.1
    else:
        ymin *= 0.9

    if ymax < 0:
        ymax *= 0.9
    elif ymax == 0:
        ymax += 0.1
    else:
        ymax *= 1.1

    # Set consistent y-limits for both subplots
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # Add grid lines for both subplots
    ax1.grid(True, which='both', axis='both')
    ax2.grid(True, which='both', axis='both')

    # Add an overall title and adjust layout
    fig.suptitle('Curve Fitting Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plots
    plt.show()


#############################################################################################################################

def plot_curve_fitting(
    actual_data: np.ndarray, 
    predicted_data: np.ndarray, 
    title: str = 'Test Data', 
    figsize: tuple = (12, 5)
) -> None:
    """
    Plots the actual vs predicted values for any dataset.

    Parameters:
    - actual_data (np.ndarray): Actual target values.
    - predicted_data (np.ndarray): Predicted values by the model.
    - title (str): Title for the plot. Defaults to 'Test Data'.
    - figsize (tuple): Size of the figure for the plot, default is (12, 5).

    Returns:
    - None: Displays the actual vs. predicted plot.
    """

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the actual vs. predicted data
    ax.plot(actual_data, color='blue', label='Actual')
    ax.plot(predicted_data, color='red', linestyle=':', label='Predicted')

    # Add legend
    ax.legend(['Actual', 'Predicted'])

    # Minor ticks for better granularity
    ax.minorticks_on()

    # Hide x-axis labels for clarity
    ax.tick_params(axis='x', labelbottom=False)

    # Determine the y-axis limits dynamically based on the data
    ymin = float(min(min(actual_data), min(predicted_data)))
    ymax = float(max(max(actual_data), max(predicted_data)))

    # Adjust ymin and ymax with scaling for better visual clarity
    if ymin < 0:
        ymin *= 1.1
    elif ymin == 0:
        ymin -= 0.1
    else:
        ymin *= 0.9

    if ymax < 0:
        ymax *= 0.9
    elif ymax == 0:
        ymax += 0.1
    else:
        ymax *= 1.1

    # Set y-limits
    ax.set_ylim(ymin, ymax)

    # Add grid lines for both x and y axes
    ax.grid(True, which='both', axis='both')

    # Set labels and title
    ax.set_ylabel('Values')
    ax.set_title(title)

    # Add an overall title and adjust layout
    fig.suptitle('Curve Fitting Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()


#############################################################################################################################

def train_val_regression(
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    actual_validation: np.ndarray, 
    predicted_validation: np.ndarray, 
    figsize: tuple = (12, 5)
) -> None:
    """
    Plots scatter plots of actual vs predicted values for both training and validation datasets
    and fits a regression line to each.

    Parameters:
    - actual_train (np.ndarray): Actual target values for the training set.
    - predicted_train (np.ndarray): Predicted values for the training set.
    - actual_validation (np.ndarray): Actual target values for the validation set.
    - predicted_validation (np.ndarray): Predicted values for the validation set.
    - figsize (tuple): Size of the figure for the plots, default is (12, 5).

    Returns:
    - None: Displays the scatter plot and regression line for training and validation data.
    """

    # Set up the figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Scatter plots for train and validation data
    ax1.scatter(actual_train, predicted_train, color='blue', facecolors='none', label='Train Data')
    ax2.scatter(actual_validation, predicted_validation, color='orange', facecolors='none', label='Validation Data')

    # Calculate min and max values for the actual train data (to be used for plotting lines)
    min_value = min(actual_train)
    max_value = max(actual_train)

    # Fit a linear regression model to the training data
    train_model = LinearRegression()
    train_model.fit(np.array(actual_train).reshape(-1, 1), predicted_train)
    a_train = train_model.coef_[0]
    b_train = train_model.intercept_

    # Fit a linear regression model to the validation data
    val_model = LinearRegression()
    val_model.fit(np.array(actual_validation).reshape(-1, 1), predicted_validation)
    a_val = val_model.coef_[0]
    b_val = val_model.intercept_

    # Plot the reference line (perfect predictions) and regression line for training data
    ax1.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')  # Perfect fit line
    ax1.plot([min_value, max_value], [a_train * min_value + b_train, a_train * max_value + b_train], 'k--', label='Fit Line')  # Fitted line

    # Plot the reference line and regression line for validation data
    ax2.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')
    ax2.plot([min_value, max_value], [a_val * min_value + b_val, a_val * max_value + b_val], 'k--', label='Fit Line')

    # Set axis labels and titles for both plots
    ax1.set_xlabel('Expected Values')
    ax1.set_ylabel('Predicted Values')
    a_train = float(a_train)
    ax1.set_title(f'Train Data: Reg. Coeff = {a_train:.2f}')
    ax1.legend()

    ax2.set_xlabel('Expected Values')
    ax2.set_ylabel('Predicted Values')
    a_val = float(a_val)
    ax2.set_title(f'Validation Data: Reg. Coeff = {a_val:.2f}')
    ax2.legend()

    # Turn on minor ticks and add grids for both subplots
    ax1.minorticks_on()
    ax1.grid(True, which='both')
    ax2.minorticks_on()
    ax2.grid(True, which='both')

    # Add an overall title and adjust layout
    fig.suptitle('Regression Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()


#############################################################################################################################

def plot_regression(
    actual_data: np.ndarray, 
    predicted_data: np.ndarray, 
    figsize: tuple = (12, 5), 
    title: str = 'Test Data'
) -> None:
    """
    Plots a scatter plot of actual vs predicted values for a dataset, and fits a regression line.

    Parameters:
    - actual_data (np.ndarray): Actual target values.
    - predicted_data (np.ndarray): Predicted values by the model.
    - figsize (tuple): Size of the figure for the plot, default is (12, 5).
    - title (str): Title for the plot. Defaults to 'Test Data'.

    Returns:
    - None: Displays the scatter plot and regression line.
    """
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plot of actual vs. predicted values
    ax.scatter(actual_data, predicted_data, color='blue', facecolors='none', label=title)

    # Calculate min and max values for the actual data
    min_value = min(actual_data)
    max_value = max(actual_data)

    # Fit a linear regression model to the data
    model = LinearRegression()
    model.fit(np.array(actual_data).reshape(-1, 1), predicted_data)
    a = model.coef_[0]  # Regression coefficient (slope)
    b = model.intercept_  # Intercept

    # Plot the reference line (perfect prediction)
    ax.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')

    # Plot the fitted regression line
    ax.plot([min_value, max_value], [a * min_value + b, a * max_value + b], 'k--', label='Fit Line')

    # Set axis labels and title
    ax.set_xlabel('Expected Values')
    ax.set_ylabel('Predicted Values')
    a = float(a)
    ax.set_title(f'{title}: Reg. Coeff = {a:.2f}')

    # Add legend
    ax.legend()

    # Turn on minor ticks and grid
    ax.minorticks_on()
    ax.grid(True, which='both')

    # Add an overall title and adjust layout
    fig.suptitle('Regression Plot')
    plt.tight_layout()

    # Show the plot
    plt.show()


#############################################################################################################################

def train_val_confusion_matrix(
    train_targets: np.ndarray, 
    train_predictions: np.ndarray, 
    validation_targets: np.ndarray, 
    validation_predictions: np.ndarray, 
    classes: list, 
    figsize: tuple = (16, 6)
) -> None:
    """
    Plots confusion matrices for both training and validation datasets.

    Parameters:
    - train_targets (np.ndarray): Actual class labels for training set.
    - train_predictions (np.ndarray): Predicted class labels for training set.
    - validation_targets (np.ndarray): Actual class labels for validation set.
    - validation_predictions (np.ndarray): Predicted class labels for validation set.
    - classes (list): List of class names/labels for display in the confusion matrix.
    - figsize (tuple): Figure size for the plot, default is (16, 6).

    Returns:
    - None: Displays the confusion matrix plots for training and validation datasets.
    """

    # Helper function to convert one-hot encoding to class indices if needed
    def convert_one_hot(labels):
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels

    # Convert one-hot encoded labels if necessary
    train_targets = convert_one_hot(train_targets)
    train_predictions = convert_one_hot(train_predictions)
    validation_targets = convert_one_hot(validation_targets)
    validation_predictions = convert_one_hot(validation_predictions)

    # Create subplots for train and validation confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute confusion matrices
    train_cm = confusion_matrix(train_targets, train_predictions)
    val_cm = confusion_matrix(validation_targets, validation_predictions)

    # Normalize the confusion matrices
    train_cm_normalized = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
    val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]

    # Calculate accuracy scores
    train_accuracy = accuracy_score(train_targets, train_predictions) * 100
    val_accuracy = accuracy_score(validation_targets, validation_predictions) * 100

    # Helper function to plot a confusion matrix
    def plot_cm(ax, cm, cm_normalized, accuracy, title, classes):
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            xlabel='Predicted Labels', ylabel='True Labels',
            title=f'{title}\nAccuracy: {accuracy:.2f}%'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate confusion matrix with counts and percentages
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.2f}%)",
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "white")

    # Plot confusion matrices for train and validation datasets
    plot_cm(ax1, train_cm, train_cm_normalized, train_accuracy, 'Train Confusion Matrix', classes)
    plot_cm(ax2, val_cm, val_cm_normalized, val_accuracy, 'Validation Confusion Matrix', classes)

    # Adjust layout and display
    fig.suptitle('Confusion Matrix Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#############################################################################################################################

def plot_confusion_matrix(data_targets: np.ndarray, data_predictions: np.ndarray, classes: list, title: str = 'Test Data', figsize: tuple = (16, 6)) -> None:
    """
    Plots a normalized confusion matrix with accuracy annotation.

    Parameters:
    - data_targets (np.ndarray): The true labels for the data, in one-hot or standard label format.
    - data_predictions (np.ndarray): The predicted labels, in one-hot or standard label format.
    - classes (list): List of class names to label the matrix axes.
    - title (str): Title of the plot, with default value 'Test Data'.
    - figsize (tuple): Figure size, defaulting to (16, 6).

    Returns:
    - None: Displays the confusion matrix plot.
    """
    
    def convert_one_hot(labels: np.ndarray) -> np.ndarray:
        """
        Converts one-hot encoded labels to single integers if needed.

        Parameters:
        - labels (np.ndarray): Array of labels, either in one-hot encoding or standard format.
        
        Returns:
        - np.ndarray: Array of labels in single integer format.
        """
        if labels.ndim == 2:
            # Converts one-hot encoded labels to integer labels by taking the argmax
            return np.argmax(labels, axis=1)
        return labels  # Returns unchanged if already in integer format

    # Convert both true and predicted labels to integer labels if in one-hot format
    data_targets = convert_one_hot(data_targets)
    data_predictions = convert_one_hot(data_predictions)
    
    # Set up the figure and axes for the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute the confusion matrix for the target and prediction labels
    cm = confusion_matrix(data_targets, data_predictions)
    
    # Normalize the confusion matrix by row (true label count)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate the overall accuracy
    accuracy = accuracy_score(data_targets, data_predictions)
    
    # Display the normalized confusion matrix as a heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)  # Add a color bar to the heatmap
    
    # Set axis labels and title with accuracy
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Labels', ylabel='True Labels',
           title=f'{title}\nAccuracy: {accuracy:.2f}%')
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Define formatting for matrix text and threshold for color change
    fmt = '.2f'
    thresh = cm.max() / 2.
    
    # Annotate each cell in the matrix with the count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f}%)",
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


#############################################################################################################################

def visualize2D_data(features: np.ndarray, targets: np.ndarray, predicted_data: np.ndarray, figsize: tuple = (8, 4), 
                     n_components: int = 2, random_state: int = 42, perplexity: int = 30, num_instance: int = 500) -> None:
    """
    Visualizes 2D or high-dimensional classification data and predicted labels.
    Uses scatter plots for 2D data and t-SNE for high-dimensional data.

    Parameters:
    - features (np.ndarray): The feature data (samples, features).
    - targets (np.ndarray): The true target labels in one-hot or standard label format.
    - predicted_data (np.ndarray): Predicted labels in one-hot format.
    - figsize (tuple): Figure size for the plot, default (8, 4).
    - n_components (int): Number of components for t-SNE if data has >2 dimensions.
    - random_state (int): Random seed for reproducibility.
    - perplexity (int): Perplexity parameter for t-SNE, default 30.
    - num_instance (int): Max number of samples to display for visualization, default 500.

    Returns:
    - None: Displays a scatter plot or t-SNE plot of the data.
    """
    
    # If the number of instances exceeds num_instance, randomly select a subset
    if features.shape[0] > num_instance:
        indices = np.random.choice(features.shape[0], num_instance, replace=False)
        features = features[indices]
        targets = targets[indices]
        predicted_data = predicted_data[indices]
    
    # Determine the number of feature dimensions
    features_num = np.size(features, axis=1)
    
    # Set up the figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Get the predicted class labels by taking the argmax across columns
    predicted_classes = np.argmax(predicted_data, axis=1)
    
    # Convert predictions to one-hot format based on predicted classes
    predicted_data_onehot = np.zeros_like(predicted_data)
    predicted_data_onehot[np.arange(len(predicted_data)), predicted_classes] = 1

    # Check if data is 2D or higher dimensional
    if features_num <= 2:
        # Plot data in 2D space if features are <= 2
        axs[0].scatter(features[:, 0], features[:, 1], c=targets[:, 0], cmap='coolwarm', marker='o', label='True Classes')
        axs[1].scatter(features[:, 0], features[:, 1], c=predicted_data_onehot[:, 0], cmap='cividis', marker='o', label='Predicted Classes')
        fig.suptitle('Visualizing Classification Performance')
    else:
        # Use t-SNE to reduce dimensionality for visualization
        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        features_tsne = tsne.fit_transform(features)
        
        # Scatter plot of t-SNE transformed data
        axs[0].scatter(features_tsne[:, 0], features_tsne[:, 1], c=targets.argmax(axis=1), cmap='coolwarm', marker='o', label='True Classes')
        axs[1].scatter(features_tsne[:, 0], features_tsne[:, 1], c=predicted_data_onehot.argmax(axis=1), cmap='cividis', marker='o', label='Predicted Classes')
        fig.suptitle('t-SNE Classification Performance')
    
    # Set titles and formatting for both subplots
    axs[0].set_title('Actual Data')
    axs[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                       right=False, labelbottom=False, labelleft=False)
    axs[0].minorticks_on()
    axs[0].grid(True, which='both', alpha=0.2, color='gray')
    
    axs[1].set_title('Predicted Data')
    axs[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                       right=False, labelbottom=False, labelleft=False)
    axs[1].minorticks_on()
    axs[1].grid(True, which='both', alpha=0.2, color='gray')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

#############################################################################################################################

def index_plot(y_test_true: np.array, y_test_pred: np.array, class_labels: list, figure_size: tuple = (12, 6)) -> None:
    """
    Plots real vs predicted labels for both training and validation data.
    
    Supports binary classification (using sigmoid output) and multiclass classification (using softmax output).
    
    Args:
        y_test_true (np.ndarray): True labels for the test data, shape (n_samples,) or (n_samples, n_classes).
        y_test_pred (np.ndarray): Predicted labels or probabilities for the test data, shape (n_samples,) or (n_samples, n_classes).
        class_labels (list of str): List of class names (e.g., ["class 1", "class 2"]).
        figsize (tuple): Figure size, defaulting to (12, 6).

    Returns:
        None
    """
    def convert_predictions_for_binary(predictions):
        """Convert continuous probabilities to binary class labels (0 or 1)."""
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        """Convert one-hot encoded labels to class indices."""
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels
    # Determine if binary classification is being used
    is_binary_classification = y_test_true.shape[1] == 1

    if is_binary_classification:
        # Convert binary predictions to 0/1 format
        y_test_pred = convert_predictions_for_binary(y_test_pred)
        # Flatten the target arrays if necessary
        y_test_true = y_test_true.flatten()
    else:
        # Convert one-hot encoded labels to indices for multi-class classification
        y_test_true = convert_one_hot(y_test_true)
        y_test_pred = convert_one_hot(y_test_pred)

    fig, axes = plt.subplots(figsize=figure_size, sharey=True)
    axes.set_title("Training Data")
    axes.set_title("Validation Data")

    # Number of samples and classes
    num_train_samples = len(y_test_true)
    num_classes = len(class_labels)
    
    # Set x-axis and y-axis properties
    x_train = np.arange(num_train_samples)
    y_ticks = np.arange(num_classes)
    
    # Training Data Plot
    axes.scatter(x_train, y_test_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axes.scatter(x_train, y_test_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axes.set_xlabel("Sample Index")
    axes.set_ylabel("Class Label")
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(class_labels)
    axes.legend()

    fig.suptitle("Real vs Predicted Labels")
    plt.tight_layout()
    plt.show()
