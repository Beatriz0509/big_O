# Modified by Tiago Fonseca (@tiagosf13), 2025

from exectimeit import timeit
import numpy as np, logging
from complexities import ALL_CLASSES, Constant, Linear, Logarithmic
from DTOs.TestDTO import TestDTO
from DTOs.ComplexityDTO import ComplexityDTO
from DTOs.ClassDTO import ClassDTO
from DTOs.InputDTO import InputDTO
from DTOs.OutputDTO import OutputDTO
from DTOs.ResponseHealthDTO import ResponseHealthDTO
from Parser import Parser
from datagen import get_mock_data, get_input_sizes
from MockInputs.MockInputs import MockInputs
import sys, subprocess, pkg_resources
from packaging import version

root_logger = logging.getLogger()


class BigO():
    """
    Big-O analysis class.

    This class is used to analyze the time complexity of a function by measuring its execution time
    for different input sizes and inferring the complexity class.
    """
    
    def __init__(self, n_measures: int = 100, n_repeats: int = 7, min_n: int = 256, max_n: int = 4096, 
                simplicity_bias: float = 1e-7, classes: list = ALL_CLASSES, verbose: bool = False):
        """
        Initializes the Big-O analysis object with specified parameters.

        Parameters:
        n_measures (int): The number of measurements to take.
        n_repeats (int): The number of times to repeat each measurement.
        min_n (int): The minimum input size for analysis.
        max_n (int): The maximum input size for analysis.
        simplicity_bias (float): Preference towards simpler models.
        classes (list): List of complexity classes to consider.
        verbose (bool): Whether to print detailed logs during analysis.
        """
        self.n_measures = n_measures
        self.n_repeats = n_repeats
        self.simplicity_bias = simplicity_bias
        self.classes = classes
        self.verbose = verbose
        self.mock_inputs = MockInputs(min_n, max_n)

    def __measure_execution_time(self, executable_function, function_definition, data_generator, mock_inputs):
        """
        Measure the execution time of a function for increasing N with multiple inputs.

        Input:
        ------
        executable_function (callable): The function to be analyzed.
        function_definition (str): The definition of the function.
        data_generator (callable): Function to generate data for input sizes.
        mock_inputs (MockInputs): Object to generate mock data inputs.

        Output:
        ------
        ns (numpy array): List of N values used as input to `data_generator`.
        execution_time (numpy array): List of execution times corresponding to each N.
        tests (list of TestDTO): List of TestDTO objects capturing input, output, and execution time.
        """
        
        # Decorate the function to measure its execution time over n_repeats
        func_timed = timeit.exectime(self.n_repeats)(executable_function)

        # Wrapper class to handle the execution of the function with multiple inputs
        class func_wrapper(object):
            def __init__(self, n):
                self.data = data_generator(function_definition, n, mock_inputs)
                self.result = None
                self.time = None
                self.variation = None

            def __call__(self):
                # Create a copy of the data to avoid side effects
                data_copy = tuple(arg.copy() if isinstance(arg, list) else arg for arg in self.data)
                # Measure execution time (with decorator) and separately capture the actual result 
                # to avoid side effects from timing instrumentation
                self.time, self.variation, self.result = func_timed(*data_copy)
                self.result = executable_function(*data_copy)  # Pass copied data to avoid mutation

        # Generate a sequence of input sizes (N)
        ns = np.linspace(self.mock_inputs.min_n, self.mock_inputs.max_n, self.n_measures).astype('int64')  # Ensure integer dtype
        execution_time = np.empty(self.n_measures)
        tests = []  # List of TestDTO objects

        progress = 0
        # Loop through each input size N and measure execution time
        for i, n in enumerate(ns):
            n = int(n)
            wrapper = func_wrapper(n)
            if self.verbose:
                root_logger.info(f"----------------------------------------------------------")
                root_logger.info(f"Generated input for N={n}: {wrapper.data}\n")
                root_logger.info(f"Execution Number: {i+1} of {self.n_measures}")
            
            wrapper()  # Call the function and time it
            # Negative timings can occur due to clock inconsistencies or measurement noise
            # Repeat the measure if negative time is detected
            while wrapper.time < 0:
                root_logger.info(f"Negative time detected for N={n}. Repeating measure...")
                wrapper()
            execution_time[i] = wrapper.time  # Store the timing

            if self.verbose:
                root_logger.info(f"Execution time for N={n}: {execution_time[i]}\n")
                root_logger.info(f"Function result for N={n}: {wrapper.result}")
            
            # Create TestDTO object for the current test
            tests.append(
                TestDTO(
                    input=InputDTO(value=str(wrapper.data)),
                    output=OutputDTO(value=str(wrapper.result)),
                    executionTime=execution_time[i]
                )
            )
            
            if self.verbose:
                progress = (i + 1) / self.n_measures * 100
                root_logger.info(f"Progress: {progress:.2f}%")
                root_logger.info(f"----------------------------------------------------------")
        
        return ns, execution_time, tests

    def __infer_big_o_class(self, ns, time):
        """
        Infer the Big-O complexity class from execution time data.

        Input:
        ------
        ns (numpy array): List of N values (input sizes).
        time (numpy array): List of execution times for each N.

        Output:
        ------
        complexities (list): List of ComplexityDTO objects for each complexity class.
        repeatTests (bool): Whether to repeat tests based on complexity fitting.
        """
        
        best_residuals = np.inf
        complexities = []
        repeatTests = False
        best = None

        # Loop through all complexity classes to find the best fit
        for class_ in self.classes:
            inst = class_()  # Instantiate the complexity class
            residuals = inst.fit(ns, time)  # Fit the model to the data

            # Create a ComplexityDTO for the current class
            complexity = ComplexityDTO(
                class_=ClassDTO(name=inst.__class__.__name__, uuid=inst.__class__.__name__.upper()),
                executionTime=time,
                meanExecutionTime=np.mean(time),
                standardDeviation=np.std(time),
                residual=residuals,
                best=False,
                status=ResponseHealthDTO(successfull=True, message="", code=200).model_dump(),
            )
            
            # Update the best class if this model has lower residuals
            if residuals < best_residuals - self.simplicity_bias:
                best_residuals = residuals
                # Repeat tests if best fit is Constant, Linear, or Logarithmic,
                # because small fluctuations in timing can significantly affect the fitting
                # for simple complexity classes.    
                repeatTests = isinstance(inst, (Constant, Linear, Logarithmic))
                best = complexity
                
            if self.verbose:
                root_logger.info('%s (r=%f)', inst, residuals)

            # Mark the best fitting complexity class as best=True in the list
            if best:
                for c in complexities:
                    if c.class_.name == best.class_.name:
                        c.best = True
                        break
            complexities.append(complexity)

        return complexities, repeatTests
    
    def __measure_and_infer(self, executable_function, function_definition, data_generator, max_retries: int = 1):
        """
        Measure execution time and infer Big-O complexity.

        Input:
        ------
        executable_function (callable): The function to analyze.
        function_definition (str): The definition of the function.
        data_generator (callable): Function to generate data.
        max_retries (int): The maximum number of retries in case of failure.

        Output:
        ------
        ns (numpy array): The list of N values used for measurements.
        times (numpy array): The execution times for each N.
        tests (list of TestDTO): A list of test results (input, output, execution time).
        complexities (list of ComplexityDTO): A list of complexity classes and their residuals.
        """
        
        repeatTests = True
        retries = 0
        while repeatTests and retries <= max_retries:
            if retries > 0 and self.verbose:
                root_logger.info(f"Repeating tests: {retries+1} of {max_retries}")
            repeatTests = False
            ns, times, tests = self.__measure_execution_time(executable_function, function_definition, data_generator, self.mock_inputs)
            complexities, repeatTests = self.__infer_big_o_class(ns, times)
            retries += 1
        return ns, times, [test.model_dump() for test in tests], [complexity.model_dump() for complexity in complexities]

    def __define_function_scopes(self, function_code: str):
        """
        Defines the function in the global scope so it can be executed.

        Input:
        ------
        function_code (str): The source code of the function.

        Output:
        ------
        global_scope (dict): The global scope containing the defined function.
        """
        global_scope = globals()
        exec(function_code, global_scope)
        return global_scope
    
    @staticmethod
    def get_function_meta_data(function_code: str):
        """
        Extract metadata from function code, including dependencies and arguments.

        Input:
        ------
        function_code (str): The source code of the function.

        Output:
        ------
        function_name (str): The name of the dependent function.
        function_definition (str): The definition of the function.
        function_args (list): List of function arguments.
        error (ResponseHealthDTO): Response containing error information if extraction fails.
        """
        error = ResponseHealthDTO(successfull=False, message="", code=400)
        parser = Parser(function_code)
        dependent_functions, independent_functions, functions = parser.getFunctionDependencies()

        # Error handling if no independent or dependent functions are detected
        if len(functions) > 1 and not independent_functions:
            error.message = "No independent functions detected."
        elif len(dependent_functions) == 0:
            error.message = "No dependent functions detected."
        elif len(dependent_functions) > 1:
            error.message = "Multiple dependent functions detected."
        elif error.message != "":
            return None, None, None, error
        else:
            try:
                # Extract function name, definition, and arguments
                function_name = dependent_functions[0] if len(functions) > 1 else functions[0].name
                function_definition = [func for func in functions if func.name == function_name][0]
                function_args = parser.getFunctionArgs(function_definition)
            except Exception as e:
                error.message = f"Error during function metadata extraction: {e}"
                return None, None, None, error

            error.code = 200
            error.message = "Function metadata extracted successfully."
            error.successfull = True
        return function_name, function_definition, function_args, error
    
    def __install_requirements(self, requirements: str) -> ResponseHealthDTO:
        """
        Install required packages using pip.

        Input:
        ------
        requirements (str): List of package requirements to install.

        Output:
        ------
        ResponseHealthDTO: Object indicating whether installation was successful.
        """
        errors = []

        for requirement in requirements:
            root_logger.info(f"Processing requirement: {requirement}")
            requirement_ = requirement.name + f"=={requirement.version}" if requirement.version else ""
            try:
                if requirement.version != "":
                    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

                    installed_version_str = installed_packages.get(requirement.name.lower())
                    if installed_version_str is not None:
                        installed_ver = version.parse(installed_version_str)
                        required_ver = version.parse(requirement.version)

                        root_logger.info(f"Found installed version: {installed_ver}")
                        if installed_ver >= required_ver:
                            root_logger.info(f"Requirement '{requirement.name}' is already installed with version '{installed_ver}'.")
                            continue  # Already installed
                    else:
                        root_logger.info(f"Requirement '{requirement.name}' not found. Proceeding with installation.")
                else:
                    # Check if the package is already installed, regardless of version
                    if requirement.name.lower() in [pkg.key.lower() for pkg in pkg_resources.working_set]:
                        root_logger.info(f"Requirement '{requirement.name}' is already installed (didn't account for version).")
                        continue  # Skip if the package is already installed
                root_logger.info(f"Installing requirement: {requirement_}")
                _ = subprocess.run(
                    ["pip", "install", f"{requirement_}"], 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                root_logger.info(f"successfully installed requirement: {requirement_}")
            except ValueError:
                errors.append(f"Invalid format: '{requirement_}'. Expected format: package==version")
            except subprocess.CalledProcessError as e:
                errors.append(f"Error installing '{requirement_}': {e.stderr}")

        if errors:
            # Remove any notice about upgrading pip
            errors = [error.split("[notice]")[0] for error in errors]

            return ResponseHealthDTO(
                successfull=False,
                message="\n".join(errors),
                code=400
            )

        return ResponseHealthDTO(
            successfull=True,
            message="All requirements installed successfully.",
            code=200
        )


    def test(self, code_to_analyze: str, requirements: str = '', data_generator: callable = get_mock_data, auto_input_sizes: bool = True):
        """
        Estimate time complexity class of a function from execution time with multiple inputs.

        Input:
        ------
        code_to_analyze (str): The source code of the function to analyze.
        requirements (str): List of package requirements to install.
        data_generator (callable): Function to generate mock input data.
        auto_input_sizes (bool): If True, automatically determine input sizes.

        Output:
        ------
        ns (numpy array): List of input sizes.
        times (numpy array): List of execution times.
        tests (list): List of TestDTO objects.
        complexities (list): List of ComplexityDTO objects.
        args (list): List of arguments for the function.
        error (ResponseHealthDTO): Response indicating success or error.
        """

        error = ResponseHealthDTO(successfull=False, message="", code=400)

        # First check if there are requirements to be installed using pip
        if requirements:
            root_logger.info("Installing requirements...")
            # Install the requirements using pip
            response = self.__install_requirements(requirements)
            if not (response and response.successfull):
                return None, None, None, None, None, response

        # Get the code metadata
        function_name, function_definition, function_args, error = self.get_function_meta_data(code_to_analyze)
        if error.successfull == False:
            return None, None, None, None, None, error
        else:
            # Define the function in the global scope
            # This is necessary to ensure that the function can be called with the correct arguments
            global_scope = self.__define_function_scopes(code_to_analyze)
            # Ensure the function is available in the global scope
            if function_name not in global_scope:
                error.message = f"Function '{function_name}' is not defined correctly."
                return None, None, None, None, None, error
            # Retrieve the function from the global scope
            function = global_scope[function_name]
            # Generate mock data for all function arguments, if not provided
            if auto_input_sizes:
                parser = Parser(code_to_analyze)

                if parser.ast_error:
                    root_logger.error(f"Error parsing function code: {parser.ast_error}", exc_info=True)
                    return None, None, None, None, None, error
                # Get the input sizes for the function arguments
                input_sizes = get_input_sizes(function_args)
                self.mock_inputs = MockInputs(input_sizes.min_n, input_sizes.max_n)
            # Wrap the function to handle multiple arguments
            try:
                def wrapped_function(*args):
                    return function(*args)
            except Exception as e:
                error.message = f"Error during function wrapping: {e}"
                return None, None, None, None, None, error
            # Increase recursion limit temporarily
            original_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(3000)

        try:
            # Measure execution time and infer complexity
            ns, times, tests, complexities = self.__measure_and_infer(
                executable_function=wrapped_function, function_definition=function_definition,
                data_generator=data_generator, max_retries=1)
            
            args =  [arg.model_dump() for arg in function_args]

            error = ResponseHealthDTO(
                successfull=True,
                message="Function complexity analyzed successfully.",
                code=200
            )

            return ns, times, tests, complexities, args, error
        except RecursionError:
            error.message = "Recursion limit exceeded. Please check the function for infinite recursion."
            return None, None, None, None, None, error
        except MemoryError:
            error.message = "Memory limit exceeded. Please check the function for infinite recursion."
            return None, None, None, None, None, error
        except OverflowError:
            error.message = "Overflow error. Please check the function for infinite recursion."
            return None, None, None, None, None, error
        except Exception as e:
            root_logger.error(f"Error during Big-O analysis: {e}", exc_info=True)
            return None, None, None, None, None, error
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(original_limit)
    
