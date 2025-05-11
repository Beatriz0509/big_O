# Modified by Tiago Fonseca (@tiagosf13), 2025

from complexities import ComplexityClass


def big_o_report(best, others):
    """ Creates a human-readable report of the output of the big_o function, 
    summarizing the best complexity class and the residuals 
    of other fitted complexity classes.

    This function generates a report comparing different complexity classes fitted 
    to execution time data, with a focus on the best-fitting class and the residuals 
    for the other candidates.

    Input:
    ------
    best -- ComplexityClass
        An object representing the complexity class that best fits the measured execution times.
        
    others -- dict
        A dictionary where the keys are `ComplexityClass` objects and the values are 
        the residuals of the fit. Residuals represent the error between the predicted and 
        actual values for the respective complexity class.

    Output:
    ------
    report -- str
        A human-readable string describing the best complexity class and the residuals of 
        the other fitted classes. The report lists the best class and the residuals for 
        the other fitted models.

    """

    # Start building the report string
    report = ""
    # Report the best complexity class
    report += 'Best : {!s:<60s} \n'.format(best)

    # Loop over the other complexity classes and their residuals
    for class_, residuals in others.items():
        # If the class is a valid ComplexityClass, format it and add it to the report
        if isinstance(class_, ComplexityClass):
            report += '{!s:<60s}    (res: {:.2G})\n'.format(class_, residuals)
    return report