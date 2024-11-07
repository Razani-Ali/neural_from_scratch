import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')


#############################################################################################################################

def train_val_loss(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    loss_validation: np.ndarray, 
    figure_size: tuple = (12, 5)
) -> None:
    """
    Plots the training and validation loss over epochs using logarithmic scale for both axes.
    
    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch to date.
    - loss_train (np.ndarray): Array of training loss values per epoch.
    - loss_validation (np.ndarray): Array of validation loss values per epoch.
    - figure_size (tuple): Size of the plot, default is (12, 5).

    Returns:
    - None: Displays the plot.
    """
    
    # Create a figure with two subplots, horizontally aligned
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Plot training loss on the first subplot with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax1.xaxis.grid(False)  # Turn off x-axis grid
    
    # Plot validation loss on the second subplot with a logarithmic y-scale
    ax2.plot(range(1, current_epoch + 1), loss_validation, color='orange')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax2.set_ylabel('Loss')  # Label the y-axis
    ax2.set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')  # Title showing the last validation loss
    ax2.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax2.xaxis.grid(False)  # Turn off x-axis grid

    # Set the y-limits for both subplots to be the same for consistency
    ymin = float(min(min(loss_train), min(loss_validation))) * 0.9  # Set lower bound to 90% of the smallest value
    ymax = float(max(max(loss_train), max(loss_validation))) * 1.1  # Set upper bound to 110% of the largest value
    ax1.set_ylim(ymin, ymax)  # Apply the same limits to training loss plot
    ax2.set_ylim(ymin, ymax)  # Apply the same limits to validation loss plot
    
    # Set an overarching title for both plots
    fig.suptitle('Live Loss Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plot
    plt.show()


#############################################################################################################################

def train_val_loss_fitting(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    loss_validation: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    actual_validation: np.ndarray, 
    predicted_validation: np.ndarray, 
    figure_size: tuple = (10, 6)
) -> None:
    """
    Plots both training/validation loss and the actual vs predicted curves for train and validation datasets.
    
    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch to date.
    - mse_train (np.ndarray): Array of Mean Squared Error (MSE) values for training data per epoch.
    - mse_validation (np.ndarray): Array of Mean Squared Error (MSE) values for validation data per epoch.
    - actual_train (np.ndarray): Actual output values for the training set.
    - predicted_train (np.ndarray): Predicted output values for the training set.
    - actual_validation (np.ndarray): Actual output values for the validation set.
    - predicted_validation (np.ndarray): Predicted output values for the validation set.
    - figure_size (tuple): Size of the plot, default is (10, 6).
    
    Returns:
    - None: Displays the loss and fitting plots.
    """
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes[0]
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(np.arange(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis with epoch info
    ax1.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing last train loss value
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Plot validation loss with a logarithmic y-scale
    ax2.plot(np.arange(1, current_epoch + 1), loss_validation, color='orange')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label x-axis with epoch info
    ax2.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax2.set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')  # Title showing last validation loss value
    ax2.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax2.xaxis.grid(False)  # Disable x-axis grid
    
    # Set the y-axis limits to the same range for both training and validation loss plots
    ymin = float(min(min(loss_train), min(loss_validation)) * 0.9)  # 90% of the minimum loss
    ymax = float(max(max(loss_train), max(loss_validation)) * 1.1)  # 110% of the maximum loss
    ax1.set_ylim(ymin, ymax)  # Apply y-axis limits to the training loss plot
    ax2.set_ylim(ymin, ymax)  # Apply y-axis limits to the validation loss plot
    
    # Extract the second row of axes for actual vs predicted curves
    ax3, ax4 = axes[1]
    
    # Plot actual and predicted values for the training set
    ax3.plot(actual_train, color='blue', label='Actual Train')  # Plot actual train data in blue
    ax3.plot(predicted_train, color='red', linestyle=':', label='Predicted Train')  # Plot predicted train data in red dashed line
    ax3.set_title('Train Data')  # Title for the training data plot
    ax3.legend()  # Add a legend to distinguish actual vs predicted
    ax3.minorticks_on()  # Enable minor ticks for clarity
    ax3.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels for this subplot
    
    # Plot actual and predicted values for the validation set
    ax4.plot(actual_validation, color='blue', label='Actual Validation')  # Plot actual validation data in blue
    ax4.plot(predicted_validation, color='red', linestyle=':', label='Predicted Validation')  # Plot predicted validation data in red dashed line
    ax4.set_title('Validation Data')  # Title for the validation data plot
    ax4.legend()  # Add a legend to distinguish actual vs predicted
    ax4.minorticks_on()  # Enable minor ticks for clarity
    ax4.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels for this subplot
    
    # Compute the global ymin and ymax for both actual and predicted values (train and validation)
    ymin = float(min(min(actual_train), min(actual_validation), min(predicted_train), min(predicted_validation)))
    # Adjust ymin based on whether it's negative, zero, or positive
    if ymin < 0:
        ymin *= 1.1
    elif ymin == 0:
        ymin -= 0.1
    else:
        ymin *= 0.9
    ymax = float(max(max(actual_train), max(actual_validation), max(predicted_train), max(predicted_validation)))
    # Adjust ymax based on whether it's negative, zero, or positive
    if ymax < 0:
        ymax *= 0.9
    elif ymax == 0:
        ymax += 0.1
    else:
        ymax *= 1.1
    
    # Apply the same y-axis limits to both the train and validation plots for consistency
    ax3.set_ylim(ymin, ymax)
    ax4.set_ylim(ymin, ymax)
    
    # Enable grid lines for both train and validation actual vs predicted plots
    ax3.grid(True, which='both', axis='both')
    ax4.grid(True, which='both', axis='both')
    
    # Set an overarching title for the figure
    fig.suptitle('Live Loss and Curve Fitting Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()


#############################################################################################################################

def train_val_loss_regression(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    loss_validation: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    actual_validation: np.ndarray, 
    predicted_validation: np.ndarray, 
    figure_size: tuple = (10, 6)
) -> None:
    """
    Plots training and validation loss along with regression plots for actual vs predicted values for both
    train and validation datasets.

    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch during training.
    - mse_train (np.ndarray): Mean Squared Error (MSE) values for the training dataset per epoch.
    - mse_validation (np.ndarray): Mean Squared Error (MSE) values for the validation dataset per epoch.
    - actual_train (np.ndarray): Actual values for the training dataset.
    - predicted_train (np.ndarray): Predicted values for the training dataset.
    - actual_validation (np.ndarray): Actual values for the validation dataset.
    - predicted_validation (np.ndarray): Predicted values for the validation dataset.
    - figure_size (tuple): Size of the figure, default is (10, 6).

    Returns:
    - None: Displays the loss and regression plots.
    """

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes[0]
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid for clarity
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Plot validation loss with a logarithmic y-scale
    ax2.plot(range(1, current_epoch + 1), loss_validation, color='orange')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax2.set_ylabel('Loss')  # Label the y-axis
    ax2.set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')  # Title showing the last validation loss
    ax2.yaxis.grid(True, which='minor')  # Enable minor grid for clarity
    ax2.xaxis.grid(False)  # Disable x-axis grid
    
    # Set y-axis limits based on minimum and maximum values of loss
    ymin = float(min(min(loss_train), min(loss_validation))) * 0.9  # Set lower limit as 90% of the minimum loss
    ymax = float(max(max(loss_train), max(loss_validation))) * 1.1  # Set upper limit as 110% of the maximum loss
    ax1.set_ylim(ymin, ymax)  # Apply limits to the training loss plot
    ax2.set_ylim(ymin, ymax)  # Apply limits to the validation loss plot

    # Extract the second row of axes for scatter plots (actual vs predicted)
    ax3, ax4 = axes[1]
    
    # Plot the scatter plot for the training set (actual vs predicted values)
    ax3.scatter(actual_train, predicted_train, color='blue', facecolors='none', label='Train Data')  # Training data points
    ax4.scatter(actual_validation, predicted_validation, color='orange', facecolors='none', label='Validation Data')  # Validation data points
    
    # Compute min and max values across the actual values to set the plotting range
    min_value = float(min(min(actual_train), min(actual_validation)))
    max_value = float(max(max(actual_train), max(actual_validation)))
    
    # Perform linear regression on the training set
    train_model = LinearRegression()
    train_model.fit(np.array(actual_train).reshape(-1, 1), predicted_train)
    a_train = train_model.coef_[0]  # Slope of the regression line
    b_train = train_model.intercept_  # Intercept of the regression line
    
    # Perform linear regression on the validation set
    val_model = LinearRegression()
    val_model.fit(np.array(actual_validation).reshape(-1, 1), predicted_validation)
    a_val = val_model.coef_[0]  # Slope of the regression line
    b_val = val_model.intercept_  # Intercept of the regression line
    
    # Plot reference line (perfect fit line) and fitted regression line for the training set
    ax3.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')  # Perfect fit line
    ax3.plot([min_value, max_value], [a_train * min_value + b_train, a_train * max_value + b_train], 'k--', label='Fit Line')  # Regression line
    
    # Plot reference line (perfect fit line) and fitted regression line for the validation set
    ax4.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')  # Perfect fit line
    ax4.plot([min_value, max_value], [a_val * min_value + b_val, a_val * max_value + b_val], 'k--', label='Fit Line')  # Regression line
    
    # Set labels and titles for the training and validation scatter plots
    ax3.set_xlabel('Expected Values')
    ax3.set_ylabel('Predicted Values')
    a_train = float(a_train)
    ax3.set_title(f'Train Data: Reg. Coeff = {a_train:.2f}')  # Title showing regression coefficient for training set
    ax3.legend()  # Show legend
    
    ax4.set_xlabel('Expected Values')
    ax4.set_ylabel('Predicted Values')
    a_val = float(a_val)
    ax4.set_title(f'Validation Data: Reg. Coeff = {a_val:.2f}')  # Title showing regression coefficient for validation set
    ax4.legend()  # Show legend
    
    # Add grid lines and minor ticks for both scatter plots
    ax3.minorticks_on()
    ax3.grid(True, which='both')
    ax4.minorticks_on()
    ax4.grid(True, which='both')
    
    # Set an overarching title for the entire figure
    fig.suptitle('Live Loss and Regression Plots')
    
    # Adjust layout to prevent overlap between titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()


#############################################################################################################################

def train_val_loss_regression_fitting(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    loss_validation: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    actual_validation: np.ndarray, 
    predicted_validation: np.ndarray, 
    figure_size: tuple = (12, 8)
) -> None:
    """
    Plots training and validation loss, actual vs predicted values, and regression fits
    for both training and validation datasets.
    
    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch of training.
    - mse_train (np.ndarray): Array of training MSE loss values.
    - mse_validation (np.ndarray): Array of validation MSE loss values.
    - actual_train (np.ndarray): Actual target values for training set.
    - predicted_train (np.ndarray): Predicted target values for training set.
    - actual_validation (np.ndarray): Actual target values for validation set.
    - predicted_validation (np.ndarray): Predicted target values for validation set.
    - figure_size (tuple): Figure size, default is (12, 6).
    
    Returns:
    - None: Displays the plots.
    """
    
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=figure_size)
    
    # First row: Training and validation loss vs. epochs (log scale)
    ax1, ax2 = axes[0]
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    ax1.yaxis.grid(True, which='minor')
    ax1.xaxis.grid(False)
    
    ax2.plot(range(1, current_epoch + 1), loss_validation, color='orange')
    ax2.set_yscale('log')
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')
    ax2.yaxis.grid(True, which='minor')
    ax2.xaxis.grid(False)
    
    # Set y-axis limits for both loss plots
    ymin = float(min(min(loss_train), min(loss_validation))) * 0.9
    ymax = float(max(max(loss_train), max(loss_validation))) * 1.1
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # Second row: Actual vs predicted values for both datasets (line plots)
    ax3, ax4 = axes[1]
    ax3.plot(actual_train, color='blue', label='Actual Train')
    ax3.plot(predicted_train, color='red', linestyle=':', label='Predicted Train')
    ax3.set_title('Train Data')
    ax3.legend()
    ax3.minorticks_on()
    ax3.tick_params(axis='x', labelbottom=False)
    
    ax4.plot(actual_validation, color='blue', label='Actual Validation')
    ax4.plot(predicted_validation, color='red', linestyle=':', label='Predicted Validation')
    ax4.set_title('Validation Data')
    ax4.legend()
    ax4.minorticks_on()
    ax4.tick_params(axis='x', labelbottom=False)
    
    # Set y-axis limits for actual vs predicted values
    ymin = float(min(min(actual_train), min(actual_validation), min(predicted_train), min(predicted_validation)))
    if ymin < 0:
        ymin *= 1.1
    elif ymin == 0:
        ymin -= 0.1
    else:
        ymin *= 0.9
    ymax = float(max(max(actual_train), max(actual_validation), max(predicted_train), max(predicted_validation)))
    if ymax < 0:
        ymax *= 0.9
    elif ymax == 0:
        ymax += 0.1
    else:
        ymax *= 1.1
    ax3.set_ylim(ymin, ymax)
    ax4.set_ylim(ymin, ymax)
    ax3.grid(True, which='both', axis='both')
    ax4.grid(True, which='both', axis='both')
    
    # Third row: Scatter plots with linear regression fits for both datasets
    ax5, ax6 = axes[2]
    ax5.scatter(actual_train, predicted_train, color='blue', facecolors='none', label='Train Data')
    ax6.scatter(actual_validation, predicted_validation, color='orange', facecolors='none', label='Validation Data')
    
    # Perform linear regression for both train and validation datasets
    min_value = float(min(min(actual_train), min(actual_validation)))
    max_value = float(max(max(actual_train), max(actual_validation)))
    
    # Train regression
    train_model = LinearRegression()
    train_model.fit(np.array(actual_train).reshape(-1, 1), predicted_train)
    a_train = train_model.coef_[0]
    b_train = train_model.intercept_
    
    # Validation regression
    val_model = LinearRegression()
    val_model.fit(np.array(actual_validation).reshape(-1, 1), predicted_validation)
    a_val = val_model.coef_[0]
    b_val = val_model.intercept_
    
    # Plot reference and fit lines for training data
    ax5.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')
    ax5.plot([min_value, max_value], [a_train * min_value + b_train, a_train * max_value + b_train], 'k--', label='Fit Line')
    
    # Plot reference and fit lines for validation data
    ax6.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')
    ax6.plot([min_value, max_value], [a_val * min_value + b_val, a_val * max_value + b_val], 'k--', label='Fit Line')
    
    # Titles and labels for regression scatter plots
    ax5.set_xlabel('Expected Values')
    ax5.set_ylabel('Predicted Values')
    a_train = float(a_train)
    ax5.set_title(f'Train Data: Reg. Coeff = {a_train:.2f}')
    ax5.legend()
    
    ax6.set_xlabel('Expected Values')
    ax6.set_ylabel('Predicted Values')
    a_val = float(a_val)
    ax6.set_title(f'Validation Data: Reg. Coeff = {a_val:.2f}')
    ax6.legend()
    
    # Minor ticks and grid for regression plots
    ax5.minorticks_on()
    ax5.grid(True, which='both')
    ax6.minorticks_on()
    ax6.grid(True, which='both')
    
    # Set an overarching title for the figure
    fig.suptitle('Live Loss, Curve Fitting, and Regression Plots')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show the figure
    plt.show()


#############################################################################################################################

def train_val_loss_confusion(
    number_of_epochs: int,
    current_epoch: int,
    Loss_train: np.ndarray,
    loss_validation: np.ndarray,
    train_targets: np.ndarray,
    train_predictions: np.ndarray,
    validation_targets: np.ndarray,
    validation_predictions: np.ndarray,
    classes: list,
    figure_size: tuple = (12, 6)
) -> None:
    """
    Plots the training/validation loss curves and confusion matrices for both datasets.

    Parameters:
    - number_of_epochs (int): Total number of training epochs.
    - current_epoch (int): The current epoch number.
    - Loss_train (np.ndarray): Training loss values over the epochs.
    - loss_validation (np.ndarray): Validation loss values over the epochs.
    - train_targets (np.ndarray): Actual labels for the training data.
    - train_predictions (np.ndarray): Predicted labels for the training data.
    - validation_targets (np.ndarray): Actual labels for the validation data.
    - validation_predictions (np.ndarray): Predicted labels for the validation data.
    - classes (list): List of class labels for the confusion matrix.
    - figure_size (tuple): Figure size, default is (12, 6).

    Returns:
    - None: Displays the loss curves and confusion matrices.
    """
    # Handle binary and multi-class classification
    def convert_predictions_for_binary(predictions):
        """ Convert continuous probabilities to binary class labels (0 or 1). """
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        """ Convert one-hot encoded labels to class indices. """
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels

    # Determine if binary classification is being used based on the shape of train_targets
    is_binary_classification = train_targets.shape[1] == 1

    if is_binary_classification:
        # Convert binary predictions to 0/1 format
        train_predictions = convert_predictions_for_binary(train_predictions)
        validation_predictions = convert_predictions_for_binary(validation_predictions)
        # Flatten the target arrays if necessary
        train_targets = train_targets.flatten()
        validation_targets = validation_targets.flatten()
        # # Binary classes: 0 and 1
        # classes = [0, 1]
    else:
        # Convert one-hot encoded labels to indices for multi-class classification
        train_targets = convert_one_hot(train_targets)
        train_predictions = convert_one_hot(train_predictions)
        validation_targets = convert_one_hot(validation_targets)
        validation_predictions = convert_one_hot(validation_predictions)

    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=figure_size)

    # Plot training loss
    axs[0, 0].plot(range(1, current_epoch + 1), Loss_train, color='blue')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title(f'Train Loss, last epoch: {Loss_train[-1]:.5f}')
    axs[0, 0].yaxis.grid(True, which='minor')
    axs[0, 0].xaxis.grid(False)

    # Plot validation loss
    axs[0, 1].plot(range(1, current_epoch + 1), loss_validation, color='orange')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')
    axs[0, 1].yaxis.grid(True, which='minor')
    axs[0, 1].xaxis.grid(False)

    # Set consistent y-limits for both loss plots
    ymin = float(min(min(Loss_train), min(loss_validation))) * 0.9
    ymax = float(max(max(Loss_train), max(loss_validation))) * 1.1
    axs[0, 0].set_ylim(ymin, ymax)
    axs[0, 1].set_ylim(ymin, ymax)

    # Compute confusion matrices for training and validation
    train_cm = confusion_matrix(train_targets, train_predictions)
    val_cm = confusion_matrix(validation_targets, validation_predictions)

    # Normalize the confusion matrices
    train_cm_normalized = train_cm.astype('float') / (train_cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    val_cm_normalized = val_cm.astype('float') / (val_cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Calculate accuracy
    train_accuracy = accuracy_score(train_targets, train_predictions)
    val_accuracy = accuracy_score(validation_targets, validation_predictions)

    # Function to plot confusion matrices
    def plot_cm(ax, cm, cm_normalized, accuracy, title, classes):
        """ Plot the confusion matrix with actual and predicted values along with accuracy. """
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted Labels', ylabel='True Labels',
               title=f'{title}\nAccuracy: {accuracy * 100:.2f}%')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells with raw counts and percentages
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.2f}%)",
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "white")

    # Plot confusion matrices
    plot_cm(axs[1, 0], train_cm, train_cm_normalized, train_accuracy, 'Train Confusion Matrix', classes)
    plot_cm(axs[1, 1], val_cm, val_cm_normalized, val_accuracy, 'Validation Confusion Matrix', classes)

    # Final adjustments to layout
    fig.suptitle('Live Loss and Confusion Matrix Plots')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#############################################################################################################################

def train_val_loss_index(
    y_train_true: np.array, y_train_pred: np.array,
    y_val_true: np.array, y_val_pred: np.array, class_labels: list,
    number_of_epochs: int, current_epoch: int,
    loss_train: np.ndarray, loss_validation: np.ndarray,
    figure_size: tuple = (15, 10)
) -> None:
    """
    Combined function to plot both real vs predicted labels for training and validation data,
    and training and validation loss over epochs.
    
    Supports binary and multiclass classification.
    """
    def convert_predictions_for_binary(predictions):
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels

    is_binary_classification = y_train_true.shape[1] == 1

    if is_binary_classification:
        y_train_pred = convert_predictions_for_binary(y_train_pred)
        y_val_pred = convert_predictions_for_binary(y_val_pred)
        y_train_true = y_train_true.flatten()
        y_val_true = y_val_true.flatten()
    else:
        y_train_true = convert_one_hot(y_train_true)
        y_train_pred = convert_one_hot(y_train_pred)
        y_val_true = convert_one_hot(y_val_true)
        y_val_pred = convert_one_hot(y_val_pred)

    fig, axes = plt.subplots(2, 2, figsize=figure_size, gridspec_kw={'height_ratios': [1, 1]})

    # Plot training and validation loss
    axes[0, 0].plot(range(1, current_epoch + 1), loss_train, color='blue')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    axes[0, 0].yaxis.grid(True, which='minor')
    axes[0, 0].xaxis.grid(False)

    axes[0, 1].plot(range(1, current_epoch + 1), loss_validation, color='orange')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')
    axes[0, 1].yaxis.grid(True, which='minor')
    axes[0, 1].xaxis.grid(False)

    ymin = float(min(min(loss_train), min(loss_validation))) * 0.9
    ymax = float(max(max(loss_train), max(loss_validation))) * 1.1
    axes[0, 0].set_ylim(ymin, ymax)
    axes[0, 1].set_ylim(ymin, ymax)

    # Plot real vs predicted labels for training and validation data
    num_train_samples = len(y_train_true)
    num_val_samples = len(y_val_true)
    num_classes = len(class_labels)

    x_train = np.arange(num_train_samples)
    x_val = np.arange(num_val_samples)
    y_ticks = np.arange(num_classes)

    axes[1, 0].scatter(x_train, y_train_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axes[1, 0].scatter(x_train, y_train_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Class Label")
    axes[1, 0].set_yticks(y_ticks)
    axes[1, 0].set_yticklabels(class_labels)
    axes[1, 0].set_title("Training Data")
    axes[1, 0].legend()

    axes[1, 1].scatter(x_val, y_val_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axes[1, 1].scatter(x_val, y_val_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_yticks(y_ticks)
    axes[1, 1].set_yticklabels(class_labels)
    axes[1, 1].set_title("Validation Data")
    axes[1, 1].legend()

    fig.suptitle("Live Loss and Index Plots")

    
    # Adjust spacing to avoid overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)  # Add more vertical space between subplots
    plt.show()

#############################################################################################################################

def train_val_loss_index_confusion(
    y_train_true: np.array, y_train_pred: np.array,
    y_val_true: np.array, y_val_pred: np.array, class_labels: list,
    number_of_epochs: int, current_epoch: int,
    loss_train: np.ndarray, loss_validation: np.ndarray,
    figure_size: tuple = (12, 8)
) -> None:
    """
    Combines loss curves, index plots for real vs. predicted labels, and confusion matrices for both datasets.
    """
    def convert_predictions_for_binary(predictions):
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        return np.argmax(labels, axis=1) if labels.ndim == 2 else labels

    is_binary_classification = y_train_true.shape[1] == 1

    if is_binary_classification:
        y_train_pred = convert_predictions_for_binary(y_train_pred).flatten()
        y_val_pred = convert_predictions_for_binary(y_val_pred).flatten()
        y_train_true = y_train_true.flatten()
        y_val_true = y_val_true.flatten()
    else:
        y_train_true = convert_one_hot(y_train_true)
        y_train_pred = convert_one_hot(y_train_pred)
        y_val_true = convert_one_hot(y_val_true)
        y_val_pred = convert_one_hot(y_val_pred)

    fig, axs = plt.subplots(3, 2, figsize=figure_size, gridspec_kw={'height_ratios': [1, 1, 2]})
    fig.suptitle("Training and Validation Loss, Real vs Predicted Labels, and Confusion Matrices", fontsize=16, y=1.02)

    # Plot training and validation loss
    axs[0, 0].plot(range(1, current_epoch + 1), loss_train, color='blue')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    axs[0, 0].yaxis.grid(True, which='minor')
    axs[0, 0].xaxis.grid(False)

    axs[0, 1].plot(range(1, current_epoch + 1), loss_validation, color='orange')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title(f'Validation Loss, last epoch: {loss_validation[-1]:.5f}')
    axs[0, 1].yaxis.grid(True, which='minor')
    axs[0, 1].xaxis.grid(False)

    ymin = float(min(min(loss_train), min(loss_validation))) * 0.9
    ymax = float(max(max(loss_train), max(loss_validation))) * 1.1
    axs[0, 0].set_ylim(ymin, ymax)
    axs[0, 1].set_ylim(ymin, ymax)

    # Plot real vs predicted labels for training and validation data
    x_train = np.arange(len(y_train_true))
    x_val = np.arange(len(y_val_true))
    y_ticks = np.arange(len(class_labels))

    axs[1, 0].scatter(x_train, y_train_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axs[1, 0].scatter(x_train, y_train_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axs[1, 0].set_xlabel("Sample Index")
    axs[1, 0].set_ylabel("Class Label")
    axs[1, 0].set_yticks(y_ticks)
    axs[1, 0].set_yticklabels(class_labels)
    axs[1, 0].set_title("Training Data")
    axs[1, 0].legend()

    axs[1, 1].scatter(x_val, y_val_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axs[1, 1].scatter(x_val, y_val_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axs[1, 1].set_xlabel("Sample Index")
    axs[1, 1].set_yticks(y_ticks)
    axs[1, 1].set_yticklabels(class_labels)
    axs[1, 1].set_title("Validation Data")
    axs[1, 1].legend()

    # Confusion matrices for training and validation
    train_cm = confusion_matrix(y_train_true, y_train_pred)
    val_cm = confusion_matrix(y_val_true, y_val_pred)
    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    val_accuracy = accuracy_score(y_val_true, y_val_pred)

    def plot_confusion_matrix(ax, cm, title, accuracy, classes):
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted', ylabel='True', title=f'{title}\nAccuracy: {accuracy * 100:.2f}%')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plot_confusion_matrix(axs[2, 0], train_cm, 'Train Confusion Matrix', train_accuracy, class_labels)
    plot_confusion_matrix(axs[2, 1], val_cm, 'Validation Confusion Matrix', val_accuracy, class_labels)

    fig.suptitle("Live Loss, Index and Confusion Matrix Plots")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.6)  # Add more vertical space between subplots
    plt.show()
