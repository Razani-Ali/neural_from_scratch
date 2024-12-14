import numpy as np
from initializers.weight_initializer import RBF_weight_init
from optimizers.set_optimizer import init_optimizer


class RBF:
    """
    Radial Basis Function (RBF) Neural Network layer.

    Parameters:
    -----------
    input_size : int
        The dimension of the input data.
    output_size : int
        The number of neurons in the RBF layer (also the dimension of the output).
    batch_size : int, optional
        The number of samples to be processed in a batch (default is 32).
    center_init_method : str, optional
        The method for initializing the centers of the RBF neurons. Options are 'random', 'zeros', and 'Kmeans' (default is 'random').
    train_center : bool, optional
        If True, the centers of the RBF neurons are trainable (default is True).
    train_var : bool, optional
        If True, the variance of the RBF neurons is trainable (default is True).
    center_distribution : str, optional
        The distribution used for initializing the centers ('normal' or 'uniform') (default is 'normal').
    data : np.ndarray, optional
        If 'Kmeans' initialization is used, this is the dataset to fit the K-means (default is None).
    var_init_method : str, optional
        The method to initialize the variances of the RBF neurons. Options are 'constant', 'average', and 'max' (default is 'average').
    var_init_const : float, optional
        The constant value used to initialize variances if var_init_method is 'constant' (default is 1).
    center_uniform_range : tuple, optional
        Range of values for uniform distribution initialization of centers (default is None).
    center_normal_var : float, optional
        The variance for normal distribution initialization of centers (default is 1).

    Attributes:
    -----------
    center : np.ndarray
        The centers of the RBF neurons.
    var : np.ndarray
        The variances (spread) of the RBF neurons.
    net : np.ndarray
        The calculated net input (distance between input and centers).
    output : np.ndarray
        The final output (activations) of the RBF neurons for the input batch.

    """
    def __init__(self, input_size: int, output_size: int, batch_size: int = 32,
                 center_init_method: str = 'random', train_center: bool = True, train_var: bool = True,
                 center_distribution: str = 'uniform', data=None, var_init_method='max', var_init_const=1,
                 center_uniform_range: tuple = (-1,1), center_normal_var: float = 1):

        # Initialize input/output sizes and training flags
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.activation = 'Guassian Kernel'

        # Initialize centers using RBF_weight_init method
        self.center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                      distribution=center_distribution, ranges=center_uniform_range,
                                      var=center_normal_var, data=data)

        # Initialize variances based on the chosen method
        if var_init_method == 'constant':
            self.var = np.zeros(output_size) + var_init_const
        elif var_init_method == 'average':
            self.var = np.mean(self.center, axis=1) / output_size
        elif var_init_method == 'max':
            self.var = np.max(self.center, axis=1) / np.sqrt(2 * output_size)

        # Initialize intermediate arrays for storing results
        self.net = np.zeros((batch_size, output_size))
        self.output = np.zeros((batch_size, output_size))

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculate the number of trainable parameters (centers and variances).

        Returns:
        --------
        int:
            The total number of trainable parameters.
        """
        params = 0
        if self.train_center:
            params += np.size(self.center)
        if self.train_var:
            params += np.size(self.var)
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculate the number of parameters (centers and variances).

        Returns:
        --------
        int:
            The total number of parameters.
        """
        return np.size(self.center) + np.size(self.var)

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of the RBF layer.

        Parameters:
        -----------
        input : np.ndarray
            Input batch of shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray:
            Output of the RBF layer (batch_size, output_size).
        """
        # Store input for backward pass
        self.input = input

        # Check if batch size is valid
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Calculate distance between input and centers
        for batch_index, input_vector in enumerate(input):
            self.net[batch_index] = np.linalg.norm((np.repeat(input_vector.reshape(1, -1),
                                                              self.output_size, axis=0) - self.center), axis=1)
            try:
                # Calculate the RBF output using Gaussian kernel
                self.output[batch_index] = np.exp(-0.5 * np.square(self.net[batch_index] / self.var))
            except:
                # Avoid division by zero
                self.output[batch_index] = np.exp(-0.5 * np.square(self.net[batch_index] / (self.var + 1e-7)))

        batch_index += 1
        return self.output[:batch_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer with specified settings.

        Parameters:
        -----------
        optimizer : str, optional
            Name of the optimizer to be used (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer.

        Returns:
        --------
        None
        """
        # Initialize the optimizer by calling an external function 'init_optimizer'
        # The optimizer is assigned based on the specified method and settings.
        # self.trainable_params() retrieves all trainable parameters for optimization.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Updates the trainable parameters (center and variance) based on gradients.

        Parameters:
        -----------
        grads : np.ndarray
            Gradients of the loss function with respect to each parameter.
        learning_rate : float, optional
            Step size for adjusting parameters (default is 1e-3).

        Returns:
        --------
        None
        """
        # Calculate the deltas for each parameter using the optimizer and gradients
        deltas = self.Optimizer(grads, learning_rate)
        
        # Initialize the starting index for parameter updates
        ind2 = 0

        # Update the center parameter if trainable
        if self.train_center:
            ind1 = ind2  # Starting index for center gradients
            ind2 += int(np.size(self.center))  # Update ending index based on center's size
            # Reshape deltas to match center's shape and apply the update
            delta_center = deltas[ind1:ind2].reshape(self.center.shape)
            self.center -= delta_center  # Update center parameter

        # Update the variance parameter if trainable
        if self.train_var:
            ind1 = ind2  # Starting index for variance gradients
            ind2 += np.size(self.var)  # Update ending index based on variance's size
            # Reshape deltas to match variance's shape and apply the update
            delta_var = deltas[ind1:ind2].reshape(self.var.shape)
            self.var -= delta_var  # Update variance parameter

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, 
                return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Computes gradients and updates parameters during backpropagation.

        Parameters:
        -----------
        error_batch : np.ndarray
            Error from the next layer, shape (batch_size, output_size).
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).
        return_error : bool, optional
            If True, returns the error propagated back to inputs (default is False).
        return_grads : bool, optional
            If True, returns computed gradients of parameters (default is False).
        modify : bool, optional
            If True, updates parameters based on gradients (default is True).

        Returns:
        --------
        dict or np.ndarray or None
            - Returns {'error_in': error_in, 'gradients': grads} if both `return_error` and `return_grads` are True.
            - Returns `error_in` if `return_error` is True and `return_grads` is False.
            - Returns `gradients` if `return_grads` is True and `return_error` is False.
        """
        # Initialize error gradient array for inputs if needed
        if return_error:
            error_in = np.zeros(self.input.shape)
        
        # Initialize gradients for center and variance if they are trainable
        grad_cen = np.zeros(self.center.shape) if self.train_center else None
        grad_var = np.zeros(self.var.shape) if self.train_var else None

        # Iterate over each error in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error = one_batch_error.ravel()
            # Compute gradient with respect to the center if it is trainable
            if self.train_center:
                # Calculate the gradient of the error with respect to the center
                # This uses the partial derivative formula for RBF centers
                grad_cen = np.diag(one_batch_error * self.output[batch_index] * self.var ** -2) @ \
                            (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.center)
            
            # Compute gradient with respect to the variance if it is trainable
            if self.train_var:
                # Calculate the gradient of the error with respect to the variance
                # This uses the partial derivative formula for RBF variances
                grad_var += (one_batch_error * self.net[batch_index] ** 2 *
                            self.var ** -3 * self.output[batch_index])
            
            # Calculate the error propagated back to the input if required
            if return_error:
                # Calculate sensitivity for input error propagation
                error_x = (one_batch_error * self.output[batch_index] * self.var ** -2).reshape((-1, 1))
                # Aggregate the propagated error based on the input dimensions
                error_in[batch_index] = (np.ones((self.input_size, self.output_size)) @ error_x).ravel() * \
                                        (np.sum(2 * self.center, axis=0).ravel() - 4 * self.input[batch_index].ravel())

        # Normalize the gradients by the batch size to get average gradients
        if self.train_center:
            grad_cen /= error_batch.shape[0]
        if self.train_var:
            grad_var /= error_batch.shape[0]

        # Combine gradients into a single array if there are gradients to be computed
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_cen is not None:
                grads = np.concatenate((grads, grad_cen.reshape((-1, 1))))  # Append center gradients
            if grad_var is not None:
                grads = np.concatenate((grads, grad_var.reshape((-1, 1))))  # Append variance gradients

        # Apply parameter updates using the computed gradients if modify is True
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return error gradients or parameter gradients based on function flags
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class TimeRBF:
    """
    Radial Basis Function (RBF) Neural Network layer for temporal data.

    Parameters:
    -----------
    time_steps : int
        Number of time steps or sequences.
    input_size : int
        The dimension of the input data.
    output_size : int
        The number of neurons in the RBF layer (also the dimension of the output).
    batch_size : int, optional
        The number of samples to be processed in a batch (default is 32).
    center_init_method : str, optional
        Method for initializing the centers of the RBF neurons. Options: 'random', 'zeros', 'Kmeans' (default is 'random').
    train_center : bool, optional
        Whether the centers of the RBF neurons are trainable (default is True).
    train_var : bool, optional
        Whether the variance (spread) of the RBF neurons is trainable (default is True).
    center_distribution : str, optional
        Distribution used for initializing the centers ('normal' or 'uniform') (default is 'uniform').
    data : Optional[np.ndarray], optional
        Dataset to fit K-means for center initialization (default is None).
    var_init_method : str, optional
        Method to initialize variances. Options: 'constant', 'average', 'max' (default is 'max').
    var_init_const : float, optional
        Constant value to initialize variances if `var_init_method` is 'constant' (default is 1).
    center_uniform_range : Tuple[float, float], optional
        Range of values for uniform distribution initialization of centers (default is (-1, 1)).
    center_normal_var : float, optional
        Variance for normal distribution initialization of centers (default is 1).

    Attributes:
    -----------
    center : np.ndarray
        The centers of the RBF neurons.
    var : np.ndarray
        The variances (spread) of the RBF neurons.
    net : np.ndarray
        The calculated net input (distance between input and centers).
    output : np.ndarray
        The final output (activations) of the RBF neurons for the input batch.
    input : np.ndarray
        Storage for the input data during computation.
    grad_cen : Optional[np.ndarray]
        Gradients for the centers, computed during backpropagation.
    grad_var : Optional[np.ndarray]
        Gradients for the variances, computed during backpropagation.
    """

    def __init__(
        self,
        time_steps: int,
        input_size: int,
        output_size: int,
        batch_size: int = 32,
        center_init_method: str = 'random',
        train_center: bool = True,
        train_var: bool = True,
        center_distribution: str = 'uniform',
        data: np.ndarray = None,
        var_init_method: str = 'max',
        var_init_const: float = 1,
        center_uniform_range: tuple = (-1, 1),
        center_normal_var: float = 1
    ):
        """
        Initializes the TimeRBF class with the provided parameters.

        Parameters:
        ----------
        time_steps : int
            Number of time steps or sequences.
        input_size : int
            Dimension of the input data.
        output_size : int
            Number of neurons in the RBF layer.
        batch_size : int, optional
            Number of samples to be processed in a batch (default is 32).
        center_init_method : str, optional
            Method for initializing centers of the RBF neurons (default is 'random').
        train_center : bool, optional
            Whether the centers are trainable (default is True).
        train_var : bool, optional
            Whether the variances are trainable (default is True).
        center_distribution : str, optional
            Distribution for initializing centers (default is 'uniform').
        data : Optional[np.ndarray], optional
            Dataset for center initialization if 'Kmeans' is chosen (default is None).
        var_init_method : str, optional
            Method to initialize variances (default is 'max').
        var_init_const : float, optional
            Constant value for variance initialization (default is 1).
        center_uniform_range : Tuple[float, float], optional
            Range for uniform distribution initialization of centers (default is (-1, 1)).
        center_normal_var : float, optional
            Variance for normal distribution initialization of centers (default is 1).
        """
        # Input and output dimensions
        self.output_size = output_size  # Number of RBF neurons
        self.input_size = input_size  # Dimension of the input
        self.batch_size = batch_size  # Number of samples in a batch
        self.time_steps = time_steps  # Number of time steps

        # Trainability flags for centers and variances
        self.train_center = train_center
        self.train_var = train_var

        # Activation function (fixed to Gaussian Kernel for RBF)
        self.activation = 'Gaussian Kernel'

        # Initialize centers using a helper method
        self.center = RBF_weight_init(
            input_size, output_size, method=center_init_method,
            distribution=center_distribution, ranges=center_uniform_range,
            var=center_normal_var, data=data
        )

        # Initialize variances using the specified method
        if var_init_method == 'constant':
            self.var = np.zeros((output_size, 1)) + var_init_const  # Constant variance
        elif var_init_method == 'average':
            self.var = np.mean(self.center, axis=1).reshape((-1, 1)) / output_size  # Average variance
        elif var_init_method == 'max':
            self.var = np.max(self.center, axis=1).reshape((-1, 1)) / np.sqrt(2 * output_size)  # Max-based variance

        # Allocate arrays to store intermediate results and inputs
        self.net = np.zeros((batch_size, time_steps, output_size, 1))  # Net input (distance to centers)
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Output of RBF neurons
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input data storage

        # Initialize gradients for centers and variances
        self.grad_cen = np.zeros(self.center.shape) if self.train_center else None
        self.grad_var = np.zeros(self.var.shape) if self.train_var else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the layer.

        This includes:
        - The number of elements in the center array if centers are trainable.
        - The number of elements in the variance array if variances are trainable.

        Returns:
        --------
        int
            Total count of trainable parameters.
        """
        params = 0  # Initialize parameter counter

        # Add the size of the center array if centers are trainable
        if self.train_center:
            params += np.size(self.center)

        # Add the size of the variance array if variances are trainable
        if self.train_var:
            params += np.size(self.var)

        return params  # Return the total number of trainable parameters
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the layer.

        This includes:
        - The total number of elements in the center array.
        - The total number of elements in the variance array.

        Returns:
        --------
        int
            Total count of all parameters.
        """
        # Sum the sizes of the center and variance arrays
        return np.size(self.center) + np.size(self.var)

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the RBF layer.

        This calculates:
        - The distance between the input data and the RBF centers.
        - The output of the RBF neurons using the Gaussian kernel.

        Parameters:
        -----------
        batch_index : int
            Current batch index.
        seq_index : int
            Current sequence index.
        input : np.ndarray
            Input data array of shape (input_size, 1).

        Returns:
        --------
        np.ndarray
            Output data array of shape (output_size, 1).
        """
        # Store the input data for the current batch and sequence
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Check if the batch size is valid
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Calculate the distance between the input and the RBF centers
        self.net[batch_index, seq_index] = np.linalg.norm(
            (np.repeat(input.reshape(1, -1), self.output_size, axis=0) - self.center),
            axis=1
        ).reshape((-1, 1))

        try:
            # Calculate the RBF neuron output using the Gaussian kernel
            self.output[batch_index, seq_index] = np.exp(
                -0.5 * np.square(self.net[batch_index, seq_index].ravel() / self.var.ravel())
            ).reshape(-1, 1)
        except ZeroDivisionError:
            # Handle division by zero by adding a small constant to the variances
            self.output[batch_index, seq_index] = np.exp(
                -0.5 * np.square(self.net[batch_index, seq_index].ravel() / (self.var.ravel() + 1e-7))
            ).reshape(-1, 1)

        # Return the final output of the RBF neurons
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the RBF layer with the specified settings.

        This function sets up the optimizer to update the trainable parameters (centers and variances)
        during training based on the specified optimizer method.

        Parameters:
        -----------
        optimizer : str, optional
            Name of the optimizer to be used (default is 'Adam').
            Examples include 'Adam', 'SGD', 'RMSprop'.
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer (e.g., learning rate, momentum).

        Returns:
        --------
        None
        """
        # Initialize the optimizer using an external initialization function
        # Pass the total number of trainable parameters to the optimizer
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the parameters of the RBF layer (centers and variances) based on gradients.

        Parameters:
        -----------
        batch_size : int
            Number of samples in the batch for gradient averaging.
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).
        grads : Optional[np.ndarray], optional
            External gradients for the parameters. If None, uses internally computed gradients.

        Returns:
        --------
        None
        """
        # If no external gradients are provided, compute gradients internally
        if grads is None:
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                # Append the gradients for centers if trainable
                if self.grad_cen is not None:
                    grads = np.concatenate((grads, self.grad_cen.reshape((-1, 1))))
                # Append the gradients for variances if trainable
                if self.grad_var is not None:
                    grads = np.concatenate((grads, self.grad_var.reshape((-1, 1))))
                grads /= batch_size  # Average gradients over the batch size

        # Compute parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize the starting index for parameter updates
        ind2 = 0

        # Update centers if trainable
        if self.train_center:
            ind1 = ind2  # Start index for center updates
            ind2 += int(np.size(self.center))  # End index for center updates
            delta_center = deltas[ind1:ind2].reshape(self.center.shape)  # Reshape deltas for center
            self.center -= delta_center  # Apply the update

        # Update variances if trainable
        if self.train_var:
            ind1 = ind2  # Start index for variance updates
            ind2 += int(np.size(self.var))  # End index for variance updates
            delta_var = deltas[ind1:ind2].reshape(self.var.shape)  # Reshape deltas for variance
            self.var -= delta_var  # Apply the update

        # Reset gradients to zero for the next iteration
        self.grad_cen = self.grad_cen * 0 if self.train_center else None
        self.grad_var = self.grad_var * 0 if self.train_var else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the gradients of trainable parameters (centers and variances).

        This method concatenates all computed gradients into a single array for optimization.

        Returns:
        --------
        Optional[np.ndarray]
            A concatenated array of all gradients for trainable parameters.
            Returns None if there are no trainable parameters.
        """
        # Initialize an empty gradient array if there are trainable parameters
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Append gradients for centers if trainable
            if self.grad_cen is not None:
                grads = np.concatenate((grads, self.grad_cen.reshape((-1, 1))))
            # Append gradients for variances if trainable
            if self.grad_var is not None:
                grads = np.concatenate((grads, self.grad_var.reshape((-1, 1))))

        return grads  # Return the concatenated gradients

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients for centers and variances during backpropagation.

        This method calculates:
        - Gradients for the centers and variances (if trainable).
        - The propagated error for the input.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        error : np.ndarray
            Error signal propagated from the subsequent layer.

        Returns:
        --------
        np.ndarray
            The propagated error for the input (shape: input_size x 1).
        """
        # Compute gradient with respect to the centers if trainable
        if self.train_center:
            # Calculate the gradient of the error with respect to the centers
            self.grad_cen = (
                np.diag(error.ravel() * self.output[batch_index, seq_index].ravel() * self.var.ravel() ** -2) @
                (np.repeat(self.input[batch_index, seq_index].reshape(1, -1), self.output_size, axis=0) - self.center)
            )

        # Compute gradient with respect to the variances if trainable
        if self.train_var:
            # Calculate the gradient of the error with respect to the variances
            self.grad_var += (
                error.ravel() * self.net[batch_index, seq_index].ravel() ** 2 *
                self.var.ravel() ** -3 * self.output[batch_index, seq_index].ravel()
            ).reshape((-1, 1))

        # Calculate sensitivity for input error propagation
        error_x = (
            error.ravel() * self.output[batch_index, seq_index].ravel() * self.var.ravel() ** -2
        ).reshape((-1, 1))

        # Aggregate the propagated error for the input dimensions
        error_in = (
            np.ones((self.input_size, self.output_size)) @ error_x
        ).ravel() * (np.sum(2 * self.center, axis=0).ravel() - 4 * self.input[batch_index, seq_index].ravel())

        return error_in.reshape((-1, 1))  # Return the propagated error for the input
