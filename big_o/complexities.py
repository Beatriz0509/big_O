"""Definition of complexity classes."""

import numpy as np


class NotFittedError(Exception):
    """Exception raised when attempting to use a model that hasn't been fitted yet."""
    pass


class ComplexityClass(object):
    """
    Abstract base class for fitting complexity classes to timing data.

    Concrete subclasses (e.g., Linear, Quadratic) implement the actual functional form.
    Provides methods for fitting, evaluating, and comparing complexity models.
    """

    #: bool: _recalculate_fit_residuals controls if the residuals value
    # returned from np.linalg.lstsq() is equivalent to the margin of
    # error of the found coefficients.
    #
    # This is normally only needed if the complexity class overrides
    # _transform_time() or _inverse_transform_time()
    _recalculate_fit_residuals = False

    def __init__(self):
        # List of parameters of the fitted function as returned by
        # the least square method np.linalg.lstsq
        self.coeff = None

    def fit(self, n, t):
        """
        Fit complexity class parameters to timing data.

        Parameters:
        ----------
        n : array-like
            Input sizes for which timing data is measured.
        t : array-like
            Execution times corresponding to each n, in seconds.

        Returns:
        -------
        residuals : float
            Sum of squared residuals of the fit.
        """
        n = np.asanyarray(n)
        t = np.asanyarray(t)

        x = self._transform_n(n)
        y = self._transform_time(t)
        coeff, residuals, _, _ = np.linalg.lstsq(x, y, rcond=-1)
        self.coeff = coeff

        if self._recalculate_fit_residuals:
            ref_t = self.compute(n)
            # SMAPE used to compute residuals when recalculating fit; 
            # ensures scale-invariant error (0 to 1)
            # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
            residuals = np.sqrt(np.sum(np.square(ref_t - t)))
        else:
            residuals = residuals[0]
        return residuals

    def compute(self, n):
        """
        Compute the fitted function value at given input size(s).

        Parameters:
        ----------
        n : array-like
            Input sizes to evaluate the fitted model on.

        Returns:
        -------
        values : array-like
            Computed/fitted times for each n.
        """
        if self.coeff is None:
            raise NotFittedError()

        # Result is linear combination of the terms with the fitted
        # coefficients
        x = self._transform_n(n)
        total = 0
        for i in range(len(self.coeff)):
            total += self.coeff[i] * x[:, i]
        return self._inverse_transform_time(total)

    def coefficients(self):
        """Return the fitted model coefficients."""
        if self.coeff is None:
            raise NotFittedError()
        return self.coeff

    def __str__(self):
        prefix = '{}: '.format(self.__class__.__name__)
        if self.coeff is None:
            return prefix + 'not yet fitted'
        return prefix + self.format_str().format(*self.coefficients()) + ' (sec)'

    # --- abstract methods

    @classmethod
    def format_str(cls):
        """
        Return a string template describing the fitted function.

        Must include one formatting argument per coefficient.
        """
        return 'FORMAT STRING NOT DEFINED'

    def _transform_n(self, n):
        """
        Define the terms of the linear combination for the complexity class.

        Output shape: (len(n), num_coefficients)
        """
        raise NotImplementedError()

    def _transform_time(self, t):
        """
        Optionally transform time for fitting.
        E.g., t -> log(t) for exponential models.
        """
        return t

    def _inverse_transform_time(self, t):
        """
        Reverse transformation applied to time.
        E.g., t -> exp(t) for exponential models.
        """
        return t

    # --- ordering support
    # Compare complexity classes based on their 'order' attribute.
    # Higher 'order' means higher computational complexity.

    def __gt__(self, other):
        return self.order > other.order

    def __lt__(self, other):
        return self.order < other.order

    def __le__(self, other):
        return (self < other) or self == other

    def __ge__(self, other):
        return (self > other) or self == other

    def __eq__(self, other):
        return self.order == other.order

    def __hash__(self):
        return id(self)


# --- Concrete implementations of the most popular complexity classes


class Constant(ComplexityClass):
    order = 10

    def _transform_n(self, n):
        return np.ones((len(n), 1))

    @classmethod
    def format_str(cls):
        return 'time = {:.2G}'


class Linear(ComplexityClass):
    order = 30

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n'


class Quadratic(ComplexityClass):
    order = 50

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n * n]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n^2'


class Cubic(ComplexityClass):
    order = 60

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n ** 3]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n^3'


class Logarithmic(ComplexityClass):
    order = 20

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), np.log(n)]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*log(n)'


class Linearithmic(ComplexityClass):
    order = 40

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n * np.log(n)]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n*log(n)'


class Polynomial(ComplexityClass):
    order = 70
    _recalculate_fit_residuals = True

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), np.log(n)]).T

    def _transform_time(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)  # Ensure it's a NumPy array
        t[t <= 0] = np.nan  # Replace non-positive values with NaN
        result = np.log(t)
        return np.nan_to_num(result, nan=0.0)

    def _inverse_transform_time(self, t):
        return np.exp(t)

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} * x^{:.2G}'

    def coefficients(self):
        """
        Return coefficients in standard form a * n^b.

        Internal representation: exp(a) * n^b
        """
        if self.coeff is None:
            raise NotFittedError()

        a, b = self.coeff
        return np.exp(a), b


class Exponential(ComplexityClass):
    order = 80
    _recalculate_fit_residuals = True

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n]).T

    def _transform_time(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)  # Ensure it's a NumPy array
        t[t <= 0] = np.nan  # Replace non-positive values with NaN
        result = np.log(t)
        return np.nan_to_num(result, nan=0.0)

    def _inverse_transform_time(self, t):
        return np.exp(t)

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} * {:.2G}^n'

    def coefficients(self):
        """
        Return coefficients in standard form a * b^n.

        Internal representation: exp(a + b*n)
        """
        if self.coeff is None:
            raise NotFittedError()

        a, b = self.coeff
        return np.exp(a), np.exp(b)


# List of all complexity classes, ordered by expected growth
ALL_CLASSES = [
    Constant, Logarithmic, Linear, Linearithmic,
    Quadratic, Cubic, Polynomial, Exponential
]
