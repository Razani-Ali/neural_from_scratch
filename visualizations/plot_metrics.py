import numpy as np
from visualizations.online_plot import train_val_loss, train_val_loss_fitting, train_val_loss_regression
from visualizations.online_plot import train_val_loss_regression_fitting, train_val_loss_confusion
from visualizations.online_plot import train_val_loss_index_confusion, train_val_loss_index

def plot_metrics(number_of_epochs: int, current_epoch: int, loss_train: list, loss_validation: list,
                 actual_train: np.ndarray, predicted_train: np.ndarray, 
                 actual_validation: np.ndarray, predicted_validation: np.ndarray, 
                 classes: list = None, **kwargs) -> None:
    """
    Plots various metrics such as loss curves, fitting plots, regression plots, or confusion matrices 
    based on the type of problem (regression/classification) and the user's preferences.

    Parameters:
    number_of_epochs (int): Total number of training epochs.
    current_epoch (int): The current epoch of training.
    loss_train (list): List containing the training loss values for each epoch.
    loss_validation (list): List containing the validation loss values for each epoch.
    actual_train (np.ndarray): Ground truth values for the training set.
    predicted_train (np.ndarray): Predicted values for the training set.
    actual_validation (np.ndarray): Ground truth values for the validation set.
    predicted_validation (np.ndarray): Predicted values for the validation set.
    **kwargs: Additional plotting options:
        - plot_loss (bool): Whether to plot loss curves (default: True).
        - plot_fitting (bool): Whether to plot fitting for regression tasks (default: False).
        - plot_reg (bool): Whether to plot regression results (default: False).
        - plot_confusion (bool): Whether to plot a confusion matrix for classification tasks (default: False).
        - classes (list, optional): List of class labels, required for confusion matrix plotting (classification tasks).

    Returns:
    None
    """

    # Retrieve the user's option to plot the loss curve (defaults to True)
    plot_loss = kwargs.get('plot_loss', True)

    # If plotting loss is enabled
    if plot_loss:

        # Retrieve optional arguments for specific types of plots (defaults to False)
        plot_fitting = kwargs.get('plot_fitting', False)  # Plot fitting (regression) option
        plot_reg = kwargs.get('plot_reg', False)          # Plot regression option
        plot_confusion = kwargs.get('plot_confusion', False)  # Plot confusion matrix for classification
        plot_index = kwargs.get('plot_index', False)  # Plot labels index for classification

        # Check if the problem type is contradictory (cannot be both regression and classification)
        if (plot_fitting or plot_reg) and plot_confusion:
            raise ValueError('Your problem cannot be both regression and classification at the same time.')

        # Case 1: Both fitting and regression plotting are enabled
        if plot_fitting and plot_reg:
            # Call regression fitting plot
            train_val_loss_regression_fitting(number_of_epochs, current_epoch, loss_train,
                                              loss_validation, actual_train[:,0], predicted_train[:,0],
                                              actual_validation[:,0], predicted_validation[:,0])
        
        # Case 2: Fitting plot is enabled (but not regression)
        elif plot_fitting and not plot_reg:
            # Call fitting plot for general cases
            train_val_loss_fitting(number_of_epochs, current_epoch, loss_train,
                                   loss_validation, actual_train[:,0], predicted_train[:,0],
                                   actual_validation[:,0], predicted_validation[:,0])
        
        # Case 3: Regression plot is enabled (but not fitting)
        elif plot_reg and not plot_fitting:
            # Call regression plot for a regression problem
            train_val_loss_regression(number_of_epochs, current_epoch, loss_train,
                                      loss_validation, actual_train[:,0], predicted_train[:,0],
                                      actual_validation[:,0], predicted_validation[:,0])

        # Case 4: Confusion matrix and index scatter for classification
        elif plot_confusion and plot_index:
            # Call index scatter for classification
            train_val_loss_index_confusion(actual_train, predicted_train, actual_validation,
                                     predicted_validation, classes, number_of_epochs,
                                     current_epoch, loss_train, loss_validation)
            
        # Case 5: Confusion matrix for classification
        elif plot_confusion and not plot_index:
            # Call confusion matrix plotting for classification
            train_val_loss_confusion(number_of_epochs, current_epoch, loss_train,
                                     loss_validation, actual_train, predicted_train,
                                     actual_validation, predicted_validation, classes)
            
        # Case 5: Index scatter for classification
        elif plot_index and not plot_confusion:
            train_val_loss_index(
                actual_train, predicted_train,
                actual_validation, predicted_validation, classes,
                number_of_epochs, current_epoch,
                loss_train, loss_validation)
        # Last case: only plot loss curves
        else:
            # Fallback to basic loss plot if no other options are valid
            train_val_loss(number_of_epochs, current_epoch, loss_train, loss_validation)
