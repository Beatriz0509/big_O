import ast, random
from MockInputs.MockBooleanInputs import MockBooleanInputs
from MockInputs.MockDateTimeInputs import MockDateTimeInputs
from MockInputs.MockDictInputs import MockDictInputs
from MockInputs.MockFloatInputs import MockFloatInputs
from MockInputs.MockIntegerInputs import MockIntegerInputs
from MockInputs.MockListInputs import MockListInputs
from MockInputs.MockSetInputs import MockSetInputs
from MockInputs.MockStringInputs import MockStringInputs
from MockInputs.MockTupleInputs import MockTupleInputs
from DTOs.MockInputsDTO import MockInputsDTO


def get_mock_class(input_type):
    """
    Choose the mock data generator class based on the type name string.

    Args:
        input_type (str): The name of the type to mock (e.g., 'int', 'list').

    Returns:
        class: Corresponding mock generator class.

    Raises:
        ValueError: If the type is not supported.
    """
    generators = {
        "int": MockIntegerInputs,
        "float": MockFloatInputs,
        "bool": MockBooleanInputs,
        "str": MockStringInputs,
        "list": MockListInputs,
        "set": MockSetInputs,
        "dict": MockDictInputs,
        "tuple": MockTupleInputs,
        "datetime": MockDateTimeInputs,
    }

    if input_type not in generators:
        raise ValueError(f"Mock data generator for input type '{input_type}' is not defined.")
    elif input_type == "Any":
        # Return a random data generator for any type
        return random.choice(list(generators.values()))
    return generators[input_type]


def get_mock_data(function_definition, n, mockInputs):
    """
    Generate a tuple of mock data values for each argument in the function.

    Args:
        function_definition (ast.arguments): Parsed AST of function arguments.
        n (int): Number of data points to generate.
        mockInputs (MockInputsDTO): Mock input configuration (e.g., ranges).

    Returns:
        tuple: Generated mock values for all function parameters.
    """
    mock_data = ()
    for arg in function_definition.args.args:
        if arg.annotation:
            mock_data += (get_mock_data_for_arg(arg.annotation, n, mockInputs),)
        else:
            mock_data += (get_mock_data_for_arg("Any", n, mockInputs),)
    return mock_data


def get_mock_data_for_arg(annotation: ast.arg, n, mockInputs):
    """
    Generate mock data for a single function argument based on its type annotation.

    Handles nested types such as list[int], dict[str, int], and tuple[int, float].

    Args:
        annotation (ast.arg): AST annotation node or string type name.
        n (int): Number of elements to generate for iterable types.
        mockInputs (MockInputsDTO): Mock configuration object.

    Returns:
        Any: A mocked value corresponding to the type annotation.
    """
    value_present = hasattr(annotation, 'value')
    slice_present = hasattr(annotation, 'slice')
    id_present = hasattr(annotation, 'id')
    dims_present = hasattr(annotation, 'dims')

    if annotation == "Any":
        # Return a random data generator for any type
        return get_mock_class("Any").get_random_data(mockInputs)
    elif not value_present and not slice_present and id_present:
        return get_mock_class(annotation.id).get_random_data(mockInputs)
    elif value_present and slice_present:
        mock_class = get_mock_class(annotation.value.id)
        inst = mock_class(mockInputs)
        # Check if it's a list or dict
        if annotation.value.id == "list":
            for i in range(n):
                inst.content.append(get_mock_data_for_arg(annotation.slice, n, mockInputs))
        elif annotation.value.id == "dict":
            for i in range(n):
                key, value = get_mock_data_for_arg(annotation.slice, n, mockInputs)
                inst.content.setdefault(key, value)
        else:
            inst.content = get_mock_data_for_arg(annotation.slice, n, mockInputs)
        return inst.content
    elif value_present and not slice_present:
        mock_class = get_mock_class(annotation.value.id)
        return mock_class.get_random_data(mockInputs)
    elif not value_present and slice_present:
        mock_class = get_mock_class(annotation.slice.id)
        return mock_class.get_random_data(mockInputs)
    elif dims_present:
        # This means its a tuple
        tup = ()
        for tp in annotation.dims:
            if hasattr(tp, 'id'):
                mock_class = get_mock_class(tp.id)
                tup += (mock_class.get_random_data(mockInputs),)
            else:
                tup += (get_mock_data_for_arg(tp, n, mockInputs),)
        return tup
    else:
        return None


def get_input_sizes(args):
    """
    Determine the minimum and maximum number of mock inputs required.

    Uses constraints defined in the mock input classes for each argument type.

    Args:
        args (list): List of argument metadata (e.g., parsed types).

    Returns:
        MockInputsDTO: Object containing min_n and max_n values for testing.
    """

    if not args:
        return MockInputsDTO(min_n=100, max_n=100000)

    mock_classes = [get_mock_class(arg.firstType.name) for arg in args]

    # Collect minimum input sizes
    mock_min_n = [mock_class.min_n for mock_class in mock_classes]
    for mock_class in mock_classes:
        if hasattr(mock_class, 'restrictions'):
            mock_min_n = mock_class.restrictions(mock_min_n)

    # Collect maximum input sizes
    mock_max_n = [mock_class.max_n for mock_class in mock_classes]
    for mock_class in mock_classes:
        if hasattr(mock_class, 'restrictions'):
            mock_max_n = mock_class.restrictions(mock_max_n)

    return MockInputsDTO(
        min_n=min(mock_min_n),
        max_n=max(mock_max_n)
    )
