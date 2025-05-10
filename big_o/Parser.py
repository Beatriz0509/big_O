import ast
from DTOs.ArgumentDTO import ArgumentDTO
from DTOs.TypeDTO import TypeDTO

class Parser:
    """ 
    This is a custom parser that analyzes Python function code.
    It parses code into an AST, extracts function definitions, arguments, types,
    and detects function dependencies.
    """

    def __init__(self, code):
        """
        Initialize the parser with source code and generate its AST.

        Args:
            code (str): Source code string to be parsed.
        """
        self.code = code
        self.ast, self.ast_error = self.__get_ast(code)

    def __get_ast(self, function_code: str):
        """
        Parses the source code into an Abstract Syntax Tree (AST).

        Args:
            function_code (str): The code to parse.

        Returns:
            tuple: (AST tree if successful, None) or (False, SyntaxError) if parsing fails.
        """
        try:
            # Parse the code into an abstract syntax tree
            tree = ast.parse(function_code)
            # Check if the first node in the tree is a FunctionDef (a Python function definition)
            return tree, None
        except SyntaxError as s:
            return False, s

    def __getFunctionCalls(self, node):
        """
        Recursively extract all function call names from a given AST node.

        Args:
            node (ast.AST): The AST node to search within.

        Returns:
            set: A set of function call names.
        """
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                calls.add(child.func.id)
            else:
                for grandchild in ast.iter_child_nodes(child):
                    calls.update(self.__getFunctionCalls(grandchild))
        return calls
    
    def getFunctionArgs(self, function_definition):
        """
        Extract argument names and types from a function definition.

        Args:
            function_definition (ast.FunctionDef): A function AST node.

        Returns:
            list[ArgumentDTO]: List of arguments with types wrapped in DTOs.
        """
        args = []
        
        for arg in function_definition.args.args:
            argType, first_type = self.getFunctionArgType(arg.annotation) if arg.annotation else ("Any", "Any")
            args.append(ArgumentDTO(
                name=arg.arg,
                firstType=TypeDTO(
                    name=first_type,
                    uuid=first_type.upper()  # This is a placeholder for the uuid
                ),
                type=argType
            ))

        return args
    
    def getFunctionArgType(self, annotation: ast.arg, first=True):
        """
        Resolve the type of a function argument from its AST annotation.

        Args:
            annotation (ast.AST): The type annotation node.
            first (bool): Whether this is the outermost type (to capture top-level container type).

        Returns:
            tuple or str: (Full type string, first container type) if first=True, else just the type string.
        """
        first_type = "Any"  # Default type

        # Handle basic types like int, str, etc.
        if not hasattr(annotation, 'value') and not hasattr(annotation, 'slice') and hasattr(annotation, 'id'):
            if first:
                first_type = annotation.id
                return annotation.id, first_type
            else:
                return annotation.id
        
        # Handle tuple types like Tuple[int, str]
        elif hasattr(annotation, 'dims'):
            str_dims = ""
            for tp in annotation.dims:
                if hasattr(tp, 'id'):
                    str_dims += f"{tp.id},"
                else:
                    str_dims += f"{self.getFunctionArgType(tp, first=False)},"
            return str_dims[:-1]  # Remove the last comma

        # Handle generic types like List[int], Dict[str, int], etc.
        if annotation.value and annotation.slice and first:
            first_type = annotation.value.id
            return f"{annotation.value.id}[{self.getFunctionArgType(annotation.slice, first=False)}]", first_type
        elif annotation.value and annotation.slice:
            return f"{annotation.value.id}[{self.getFunctionArgType(annotation.slice, first=False)}]"
        elif annotation.value and not annotation.slice:
            if first:
                first_type = annotation.value.id
                return str(annotation.value.id), first_type
            else:
                return str(annotation.value.id)
        elif not annotation.value and annotation.slice:
            if first:
                first_type = annotation.slice.id
                return str(annotation.slice.id), first_type
            else:
                return str(annotation.slice.id)
        else:
            return "Any", "Any"
        
    def __getFunctionDefinitions(self):
        """
        Retrieve all function definitions from the AST.

        Returns:
            list[ast.FunctionDef]: List of function definition nodes.
        """
        return [node for node in self.ast.body if isinstance(node, ast.FunctionDef)] 
    
    def getFunctionDependencies(self):
        """
        Analyze function dependencies by checking which functions call each other.

        Returns:
            tuple: 
                - dependent_functions (list): Functions that are not called by others.
                - independent_functions (list): Functions that are called within other functions.
                - functions (list[ast.FunctionDef]): All function definitions found.
        """
        dependent_functions = []
        independent_functions = []

        functions = self.__getFunctionDefinitions()
        function_names = [function.name for function in functions]

        for function in functions:
            # Remove the function that is being defined
            functionCalls = [call for call in self.__getFunctionCalls(function) if call != function.name]

            # Get the function calls that are in functions
            common_calls = [call for call in functionCalls if call in function_names]
            independent_functions.extend(common_calls)
        
        dependent_functions = [func for func in function_names if func not in independent_functions]

        return dependent_functions, independent_functions, functions
