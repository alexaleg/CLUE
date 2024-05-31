import re

from functools import cached_property, reduce

from pyparsing import (
    Literal,
    ParseException,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    Suppress,
)

import sympy
from sympy import QQ, oo
from clue.field import RR
RR = RR()
from .linalg import SparseVector
from .nual import NualNumber

# ------------------------------------------------------------------------------


def to_rational(s):
    denom = 1
    extra_num = 1
    if ("E" in s) or ("e" in s):
        s, exp = re.split("[Ee]", s)
        if exp[0] == "-":
            denom = 10 ** (-int(exp))
        else:
            extra_num = 10 ** (int(exp))

    frac = s.split(".")
    if len(frac) == 1:
        return QQ(int(s) * extra_num, denom)
    return QQ(int(frac[0] + frac[1]) * extra_num, denom * 10 ** (len(frac[1])))


# ------------------------------------------------------------------------------


class SparsePolynomial(object):
    r"""
    Simplistic class for representing polynomials with sparse exponent vectors

    Input
        ``varnames`` - list of names of the variables for the polynomial.
        ``domain`` - ambient space (in sympy format) for the coefficients of the polynomial
        ``data`` - dictionary from monomials to coefficients. Monomials are encoded as
            tuples of pairs (index_of_variable, exponent) with only nonzero exponents stored

    Examples::

        >>> from clue.rational_function import *
        >>> from sympy import parse_expr, simplify
        >>> poly = "a * (3 * a + b) - 8.5 * (a + b)**5 - 3 * c * b * (c - a)"
        >>> parsed = parse_expr(poly)
        >>> sp = SparsePolynomial.from_string(poly, ["a", "b", "c"])
        >>> simplify(parse_expr(str(sp)) - parsed) == 0
        True
        >>> poly = "(a + b + c**2)**5 - 3 * a + b * 17 * 19 * 0.5"
        >>> parsed = parse_expr(poly)
        >>> sp = SparsePolynomial.from_string(poly, ["a", "b", "c"])
        >>> simplify(parse_expr(str(sp)) - parsed) == 0
        True
    """

    def __init__(self, varnames, domain=QQ, data=None, cast=True):
        self._varnames = varnames
        self._domain = domain
        if cast:
            self._data = (
                dict()
                if data is None
                else {
                    key: domain.convert(data[key])
                    for key in data
                    if data[key] != domain.zero
                }
            )
        else:
            self._data = (
                dict()
                if data is None
                else {key: data[key] for key in data if data[key] != 0}
            )

    def dataiter(self):
        return self._data.items()

    @property
    def size(self):
        return len(self._data)

    @property
    def domain(self):
        return self._domain

    @property
    def gens(self):
        return self._varnames.copy()

    @property
    def monomials(self):
        r"""
        Monomials that have a non-zero coefficient in this polynomial.

        This method returns a tuple with the monomials that have a non-zero coefficient
        in this polynomial. This whole polynomial can be retrieved from these monomials
        and the coefficients obtained by :func:`coefficients`.

        Output
            A tuple with :class:`SparsePolynomial` that are the monomials forming this polynomial.

        Examples::

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> p = SparsePolynomial.from_string("1 + x/2 + 3*y + 5*x*y", ['x','y'])
            >>> p.monomials
            (1, x, y, x*y)
            >>> SparsePolynomial.from_const(1, ["x", "y"]).monomials
            (1,)

        This method return an empty tuple if no monomial is contained, i.e., the polynomial
        is equal to zero::

            >>> SparsePolynomial(["x", "y"], QQ).monomials
            ()
            >>> SparsePolynomial.from_const(0, ["x", "y"]).monomials
            ()

        The same polynomial can be obtained using together the method :func:`coefficients`::

            >>> n = len(p.dataiter())
            >>> p == sum([p.coefficients[i]*p.monomials[i] for i in range(n)], SparsePolynomial(p.gens, p.domain))
            True
        """
        return tuple(
            [
                SparsePolynomial.monomial(term[0], self._varnames, self.domain)
                for term in self.dataiter()
            ]
        )

    @property
    def coefficients(self):
        r"""
        Coefficients of this polynomial.

        This method returns a tuple with the coefficients that appear
        in this polynomial. This whole polynomial can be retrieved from these coefficients
        and the monomials obtained by :func:`monomials`.

        Output
            A tuple with elements in ``self.domain`` that are the coefficients forming this polynomial.

        Examples::

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> p = SparsePolynomial.from_string("1 + x/2 + 3*y + 5*x*y", ['x','y'])
            >>> print(p.coefficients)
            (MPQ(1,1), MPQ(1,2), MPQ(3,1), MPQ(5,1))
            >>> SparsePolynomial.from_const(10, ["x", "y"]).coefficients
            (MPQ(10,1),)

        This method return an empty tuple if no monomial is contained, i.e., the polynomial
        is equal to zero::

            >>> SparsePolynomial(["x", "y"], QQ).coefficients
            ()
            >>> SparsePolynomial.from_const(0, ["x", "y"]).coefficients
            ()

        The same polynomial can be obtained using together the method :func:`monomials`::

            >>> n = len(p.dataiter())
            >>> p == sum([p.coefficients[i]*p.monomials[i] for i in range(n)], SparsePolynomial(p.gens, p.domain))
            True
        """
        return tuple([el[1] for el in self.dataiter()])

    @property
    def content(self):
        r"""
        Content of a polynomial.

        For a polynomial, the content is the greatest common divisor of its coefficients. This
        computation is performed on the domain of ``self`` and may have interesting behaviors
        when the domain is a field.

        Output
            An element in ``self.domain`` that is the GCD of the coefficients of ``self``.

        Examples::

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> p = SparsePolynomial.from_string("15*x + 6", ["x"])
            >>> p.content
            3
            >>> p = SparsePolynomial.from_string("15*x + 6", ["x"])//SparsePolynomial.from_string("7", ["x"])
            >>> p.content
            3/7

        The constants always return their value::

            >>> p = SparsePolynomial.from_const(QQ(5)/QQ(7), ['x'])
            >>> p.content
            5/7
            >>> p = SparsePolynomial.from_string("15", ['x'])
            >>> p.content
            15
            >>> p = SparsePolynomial(['x'])
            >>> p.content
            0
        """
        return sympy.polys.polytools.gcd(self.coefficients)

    @property
    def constant_term(self):
        r"""
        Constant coefficient of a Sparse polynomial.

        This property is the value of the constant term of the polynomial.
        This is the coefficients associated with the monomial `1`. In terms
        of the current implementation, this is the coefficient that can
        be obtained with ``self._data.get((), 0)``.

        Output
            An element in ``self.domain`` that is the coefficient of the monomial `1`.

        Examples::

            >>> from clue.rational_function import *
            >>> sp = SparsePolynomial.from_string("x*y*z + x*6 - 10", ['x','y','z'])
            >>> sp.constant_term
            MPQ(-10,1)
            >>> sp = SparsePolynomial.from_string("x - y", ['x','y'])
            >>> sp.constant_term
            MPQ(0,1)
            >>> sp = SparsePolynomial.from_const(13, ['x','y','z'])
            >>> sp.constant_term
            MPQ(13,1)
            >>> sp = SparsePolynomial(['x']) # zero polynomial
            >>> sp.constant_term
            MPQ(0,1)

        This property can also be obtained via :func:`ct`::

            >>> sp.ct
            MPQ(0,1)
        """
        return self._data.get((), self.domain.zero)

    ct = constant_term  #: alias for the constant term property

    @property
    def linear_components(self):
        r"""
        Linear components and coefficients from this polynomial.

        This method returns a set of functions (:class:`SparsePolynomial`)
        that are linearly independent w.r.t. the domain of this
        polynomial (see :func:`domain`) and a list of coefficients
        that allows to get the same polynomial again.

        For a polynomial, this is the same as the set of monomials
        of the polynomial and the list of coefficients.

        Output
            Two tuples `T, C` such that ``self == sum(C[i]*T[i] for i in range(len(T)))``.

        Examples

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> x = SparsePolynomial(["x", "y"], QQ, {tuple([(0,1)]): 1})
            >>> y = SparsePolynomial(["x", "y"], QQ, {tuple([(1,1)]): 1})
            >>> one = SparsePolynomial(["x", "y"], QQ, {(): 1})
            >>> p = one + x//(2*one) + (3*one)*y + (5*one)*x*y
            >>> print(p.linear_components)
            ((1, x, y, x*y), (MPQ(1,1), MPQ(1,2), MPQ(3,1), MPQ(5,1)))
        """
        return self.monomials, self.coefficients

    def degree(self, var_name=None):
        r"""
        Obtain the degree of this polynomial (maybe with respect to some variable)

        This method computes the degree of a :class:`SparsePolynomial`. If the input is
        ``None``, the result is the total degree of the polynomial. Otherwise, we
        return the degree of ``self`` w.r.t. the given variable.

        Input
            ``var_name`` - name (string) of the variable to compute the degree.

        Output
            If ``var_name`` is ``None``, the total degree is returned, otherwise
            we return the degree of ``self`` w.r.t. ```var_name``.

        Examples::

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> x = SparsePolynomial(["x", "y"], QQ, {tuple([(0,1)]): 1})
            >>> x.degree()
            1
            >>> x.degree('x')
            1
            >>> x.degree('y')
            0
            >>> p = SparsePolynomial.from_string("1 + x/2 + 3*y + 5*x*y", ['x','y'])
            >>> p.degree()
            2
            >>> p.degree('x')
            1
            >>> p.degree('y')
            1

        In case a variable that does not exists is given, a ValueError is raised::

            >>> p.degree('z')
            Traceback (most recent call last):
            ...
            ValueError: the variable z is not valid for this polynomial

        By convention, if the polynomial is zero, we stablish the degree to be ``oo``
        which is the infinity in :mod:`sympy`.

            >>> zero = SparsePolynomial(['x','y','z'],QQ)
            >>> zero.degree()
            oo
            >>> zero.degree('x')
            oo
            >>> zero.degree('a')
            Traceback (most recent call last):
            ...
            ValueError: the variable a is not valid for this polynomial
            >>> SparsePolynomial.from_const(0, ['x','y']).degree()
            oo
        """
        if var_name is None:
            degree_fun = lambda monomial: sum(el[1] for el in monomial)
        elif var_name not in self._varnames:
            raise ValueError(
                "the variable %s is not valid for this polynomial" % var_name
            )
        else:

            def degree_fun(monomial):
                var_index = self._varnames.index(var_name)
                red = [el for el in monomial if el[0] == var_index]
                if len(red) == 0:
                    return 0
                return red[0][1]

        if self.is_zero():
            return oo

        return max([degree_fun(term[0]) for term in self.dataiter()])

    def variables(self, as_poly=False):
        r"""
        Variables that actually appear in the polynomial.

        This method computes the variables that appear in the :class:`SparsePolynomial`
        or, equivalently, the variables that have a positive degree.

        Input
            ``as_poly`` - (optional) decides whether to return the names of the
            variables (``False``) or the variables as :class:`SparsePolynomial` (``True``)


        Output
            A tuple with the variables that have positive degree.

        Examples::

            >>> from sympy import QQ
            >>> from clue.rational_function import SparsePolynomial
            >>> x = SparsePolynomial(["x", "y", "z"], QQ, {tuple([(0,1)]): 1})
            >>> x.variables()
            ('x',)
            >>> x.variables(True)
            (x,)
            >>> y = SparsePolynomial(["x", "y", "z"], QQ, {tuple([(1,1)]): 1})
            >>> one = SparsePolynomial(["x", "y", "z"], QQ, {(): 1})
            >>> p = one + x//(2*one) + (3*one)*y + (5*one)*x*y
            >>> p.variables()
            ('x', 'y')
            >>> p.variables(True)
            (x, y)

        The constant polynomials provide an empty tuple as result::

            >>> one.variables()
            ()
            >>> SparsePolynomial(["x", "y", "z"], QQ).variables() # checking the zero polynomial
            ()
        """
        var_index = list(
            set(sum([[var[0] for var in term[0]] for term in self.dataiter()], []))
        )

        result = [self._varnames[var_index[i]] for i in range(len(var_index))]
        if as_poly:
            result = [
                SparsePolynomial.var_from_string(name, self._varnames)
                for name in result
            ]

        return tuple(result)

    # --------------------------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, SparsePolynomial):
            if other in self.domain:
                other = SparsePolynomial.from_const(other, self._varnames, self.domain)
            elif isinstance(other, str):
                other = SparsePolynomial.from_string(other, self._varnames, self.domain)
            else:
                return NotImplemented

        result = SparsePolynomial(self._varnames, self.domain)
        resdata = dict()
        for m, c in self._data.items():
            sum_coef = c + other._data.get(m, self.domain.zero if self.domain != RR else 0.0)
            if sum_coef != 0:
                resdata[m] = sum_coef

        for m, c in other._data.items():
            if m not in self._data:
                resdata[m] = c
        result._data = resdata
        return result

    def __radd__(self, other):
        return self.__add__(other)

    # --------------------------------------------------------------------------

    def __iadd__(self, other):
        if not isinstance(other, SparsePolynomial):
            if other in self.domain:
                other = SparsePolynomial.from_const(other, self._varnames, self.domain)
            elif isinstance(other, str):
                other = SparsePolynomial.from_string(other, self._varnames, self.domain)
            else:
                return NotImplemented

        for m, c in other._data.items():
            sum_coef = c + self._data.get(m, self.domain.zero)
            if sum_coef != 0:
                self._data[m] = sum_coef
            else:
                if m in self._data:
                    del self._data[m]
        return self

    # --------------------------------------------------------------------------

    def __neg__(self):
        return SparsePolynomial(
            self._varnames, self.domain, {m: -c for m, c in self._data.items()}
        )

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __isub__(self, other):
        self += -other
        return self

    # --------------------------------------------------------------------------

    def __eq__(self, other):
        r"""
        Equality method for :class:`SparsePolynomial`.

        Input
            ``other`` - object to compare with ``self``.

        Output
            ``True`` if ``other`` is a sparse polynomial (or a constant in the domain of ``self``,
            see property :func:`domain`) and has the same data as ``self``.

        Examples::

            >>> from clue.rational_function import *
            >>> sp1 = SparsePolynomial.from_string("2*x**23 + 4", ['x'])
            >>> sp2 = SparsePolynomial.from_string("2*x**23 + 4", ['x'])
            >>> sp1 is sp2
            False
            >>> sp1 == sp2
            True

        This method also recognizes ``other`` to be elements in ``self.domain``::

            >>> sp = SparsePolynomial.from_string("1 + 6", ['x'])
            >>> sp == 7
            True

        This equality check do not distinguish between variable names: the order
        that is used in self.gens is critical here::

            >>> sp1 = SparsePolynomial.from_string("x + 2*y", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("2*x + y", ['y','x'])
            >>> sp1 == sp2
            True
        """
        if not isinstance(other, SparsePolynomial):
            if isinstance(other, RationalFunction):
                return self * other.denom == other.numer
            elif other in self.domain:
                other = SparsePolynomial.from_const(other, self._varnames, self.domain)
            elif isinstance(other, str):
                other = SparsePolynomial.from_string(other, self._varnames, self.domain)
            else:
                return False
        if self._data != other._data:
            return False
        else:
            return True

    def __hash__(self) -> int:
        r"""Method to get the hash of a SparsePolynomial"""
        return hash(tuple(self._data)) * hash(
            tuple(SparsePolynomial.from_const(1, self.variables(), self.domain)._data)
        )

    # --------------------------------------------------------------------------

    def __mul__(self, other):
        """
        Multiplication by a scalar or another polynomial
        For polynomials we use slow quadratic multiplication as needed only for parsing
        """
        if type(other) == SparsePolynomial:
            result = SparsePolynomial(self._varnames, self.domain)
            for ml, cl in self._data.items():
                for mr, cr in other._data.items():
                    dictl = dict(ml)
                    dictr = dict(mr)
                    for varind, exp in dictr.items():
                        if varind in dictl:
                            dictl[varind] += exp
                        else:
                            dictl[varind] = exp
                    m = tuple([(v, dictl[v]) for v in sorted(dictl.keys())])
                    if m in result._data:
                        result._data[m] += cl * cr
                        if result._data[m] == 0:
                            del result._data[m]
                    else:
                        result._data[m] = cl * cr
            return result
        else:
            result = SparsePolynomial(self._varnames, self.domain)
            if other != 0:
                other = self.domain.convert(other)
                for m, c in self._data.items():
                    result._data[m] = c * other
            return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        return self.exp(power)

    # --------------------------------------------------------------------------

    def __floordiv__(self, other):
        r"""
        Exact division implemented with SymPy.

        This method implements the magic logic behind ``//``. This method computes
        the *exact division* between two :class:`SparsePolynomials`. This, if we consider
        an Euclidean division of the type

        .. MATH::

            A(\mathbf{x}) = q(\mathbf{x})B(\mathbf{x}) + r(\mathbf{x}),

        then this method returns the polynomial `q(\mathbf{x})`. It is based on the
        SymPy computation of the exact division. The remainder can be obtained in a similar
        way with the magic syntax ``%``.

        Input
            ``other`` - polynomial that will be the quotient (i.e., `B(\mathbf{x})`)

        Output
            The exact division polynomial `q(\mathbf{x})`.

        Examples::

            >>> from clue.rational_function import *
            >>> sp1 = SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("x+1", ['x'])
            >>> sp1//sp2
            2 + x**2 + 2*x

        **Warning:** when the variables of the divisor are not included in the variables of the dividend,
        some weird phenomena could happen::

            >>> sp1 = SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("x+y", ['x','y'])
            >>> sp1//sp2
            4 + x**2 + y**2 - 3*y + 3*x - x*y
            >>> sp3 =  SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x'])
            >>> sp3 == sp1
            True
            >>> sp3//sp2
            2 + x**2 + 2*x
        """
        if self.is_zero():
            return SparsePolynomial.from_const(0, self._varnames, self.domain)
        elif self == other:
            return SparsePolynomial.from_const(1, self._varnames, self.domain)
        elif self.is_constant() and other.is_constant():
            return SparsePolynomial.from_const(
                self.ct / other.ct, self._varnames, self.domain
            )

        ## General case (self != other and 0)
        R = self.get_sympy_ring()
        num = R(self.get_sympy_dict()).as_expr()
        denom = R(other.get_sympy_dict()).as_expr()
        quo = R(sympy.polys.polytools.quo(num, denom))
        return SparsePolynomial.from_sympy(quo)

    def __mod__(self, other):
        r"""
        Remainder computation implemented with SymPy.

        This method implements the magic logic behind ``%``. This method computes
        the *remainder* between two :class:`SparsePolynomials`. This, if we consider
        an Euclidean division of the type

        .. MATH::

            A(\mathbf{x}) = q(\mathbf{x})B(\mathbf{x}) + r(\mathbf{x}),

        then this method returns the polynomial `r(\mathbf{x})`. It is based on the
        SymPy computation of the exact division. The remainder can be obtained in a similar
        way with the magic syntax ``%``.

        Input
            ``other`` - polynomial that will be the quotient (i.e., `B(\mathbf{x})`)

        Output
            The exact division polynomial `r(\mathbf{x})`.

        Examples::

            >>> from clue.rational_function import *
            >>> sp1 = SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("x+1", ['x'])
            >>> sp1%sp2
            3

        **Warning:** when the variables of the divisor are not included in the variables of the dividend,
        some weird phenomena could happen::

            >>> sp1 = SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("x+y", ['x','y'])
            >>> sp1%sp2
            5 - y**3 - 4*y + 3*y**2
            >>> sp3 =  SparsePolynomial.from_string("x**3 + 3*x**2 + 4*x + 5", ['x'])
            >>> sp3 == sp1
            True
            >>> sp3%sp2
            3
        """
        R = self.get_sympy_ring()
        num = R(self.get_sympy_dict()).as_expr()
        denom = R(other.get_sympy_dict()).as_expr()
        if num.is_zero:
            return SparsePolynomial.from_string("0", self._varnames, self.domain)
        elif num == denom:
            return SparsePolynomial.from_const(1, self._varnames, self.domain)
        elif denom == 1:
            return self
        quo = R(sympy.polys.polytools.rem(num, denom))
        return SparsePolynomial.from_sympy(quo)

    def __truediv__(self, other):
        r"""
        True division for sparse polynomials.

        This method implements the magic logic behind ``/``. This method computes
        the *true division* between two :class:`SparsePolynomials`. This will
        create :class:`RationalFunction` when the exact division (see :func:`__floordiv__`
        and :func:`__mod__`) is different to the true division.

        Input
            ``other`` - polynomial that will be the quotient

        Output
            The true division polynomial. If ``self % other == 0`` the output is a
            :class:`SparsePolynomial`, otherwise a :class:`RationalFunction`.

        Examples::

            >>> from clue.rational_function import *
            >>> p1 = SparsePolynomial.from_string("x^2 + 2*x*y + y^2", ['x','y'])
            >>> p2 = SparsePolynomial.from_string("x+y", ['x','y'])
            >>> p1/p2
            x + y
            >>> isinstance(p1/p2, SparsePolynomial)
            True
            >>> p2/p1
            RationalFunction(1, x + y)
            >>> p1/"x"
            RationalFunction(x**2 + 2*x*y + y**2, x)
            >>> p1/10
            1/10*x**2 + 1/5*x*y + 1/10*y**2
            >>> isinstance(p1/10, SparsePolynomial)
            True

        """
        if not isinstance(other, SparsePolynomial):
            if other in self.domain:
                other = SparsePolynomial.from_const(other, self._varnames, self.domain)
            elif isinstance(other, str):
                other = SparsePolynomial.from_string(other, self._varnames, self.domain)
            else:
                return NotImplemented

        if other.is_constant():
            return SparsePolynomial(
                self._varnames,
                self.domain,
                {k: self._data[k] / other.ct for k in self._data},
            )
        else:
            if (
                self.domain.is_Exact and (self % other).is_zero()
            ):  ## Keeping SparsePolynomial if the division is exact
                return self // other
            return RationalFunction(self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, SparsePolynomial):
            if other in self.domain:
                other = SparsePolynomial.from_const(other, self._varnames, self.domain)
            elif isinstance(other, str):
                other = SparsePolynomial.from_string(other, self._varnames, self.domain)
            else:
                return NotImplemented

        return other / self

    # --------------------------------------------------------------------------

    def eval(self, **values):
        r"""
        Method to evaluate a polynomial.

        This method evaluates a polynomial substituting its variables by given values simultaneously.
        Currently, the only valid input for the values are elements contained in
        ``self.domain``.

        TODO: include evaluation with elements that are :class:`SparsePolynomial`.
        TODO: implement a wider evaluation with generic entries?

        Input
            values - dictionary containing the names fo the variables to be evaluated and the values to plug-in.

        Output
            the evaluated polynomial in the given values.

        Examples::

            >>> from clue.rational_function import *
            >>> sp = SparsePolynomial.from_string("x**2*z + y", ['x','y','z'])
            >>> sp.eval(x=2)
            4*z + y
            >>> sp.eval(x = 1/QQ(2), y = 3, z = 2)
            7/2
            >>> sp.eval(y=0)
            x**2*z
            >>> sp.eval()
            x**2*z + y
            >>> sp.eval(x=0, y=0)
            0
        """
        # analyzing the values given
        rem_variables = [el for el in self.gens]
        for el in values:
            if el in rem_variables:
                rem_variables.remove(el)

        rem_variables_indices = [self._varnames.index(el) for el in rem_variables]
        values = {
            self._varnames.index(el): (
                values[el].change_base(self.domain)
                if isinstance(values[el], NualNumber)
                else self.domain.convert(values[el])
            )
            for el in values
            if el in self._varnames
        }
        ## Here `rem_variables_indices` contains the indices of the variables remaining in the evaluation
        ## and values `values` contains a dictionary index -> value (instead of the name of the variable)

        new_data = {}
        for monomial, coefficient in self._data.items():
            ## cleaning from monomial the variables evaluated
            new_monomial = tuple(
                [
                    (rem_variables_indices.index(v), e)
                    for (v, e) in monomial
                    if v in rem_variables_indices
                ]
            )
            ## computing the new coefficient for the new monomial
            value = reduce(
                lambda p, q: p * q,
                [coefficient]
                + [
                    values[v] ** e
                    for v, e in monomial
                    if not v in rem_variables_indices
                ],
            )

            ## adding the new monomial
            if not new_monomial in new_data:
                new_data[new_monomial] = self.domain.zero
            new_data[new_monomial] += value

        ## Returning the resulting polynomial (only remaining variables appear in the polynomial)
        return SparsePolynomial(rem_variables, self.domain, new_data, cast=False)

    def subs(self, to_subs=None, **values):
        r"""
        More generic method that allows to substitute in a SparsePolynomial all the appearing variables.

        Examples::

            >>> from clue.rational_function import *
            >>> p = SparsePolynomial.from_string("x**2 + y", ["x", "z", "y"])
            >>> q1 = SparsePolynomial.from_string("x**2 + z", ["x", "z"])
            >>> q2 = SparsePolynomial.from_string("z-x", ["x", "z"])
            >>> p.subs([("x", q1), ("y", q2)])
            x**4 + 2*x**2*z + z**2 + z - x
            >>> p.subs(x=q1, y=q2)
            x**4 + 2*x**2*z + z**2 + z - x
        """
        if isinstance(to_subs, (list, tuple)):
            if len(values) > 0:
                raise TypeError(
                    "The method subs works either with a list of substitutions or with a dictionary, not mixed"
                )
            values = {str(var): val for (var, val) in to_subs}

        if any(not v in values for v in self.variables()):
            raise TypeError(
                f"Not enough variables were given for substitution. Required {self.variables()}, given {list(values.keys())}"
            )

        # we assume the user has provided everything of the same type
        to_sub = {self._varnames.index(k): v for (k, v) in values.items()}
        prod = lambda g: reduce(lambda p, q: p * q, g, 1)
        return sum(
            (c * prod(to_sub[v] ** e for (v, e) in m)) for (m, c) in self._data.items()
        )

    @cached_property
    def numerical_evaluator(self):
        return eval(f"lambda {','.join(self._varnames)}: {str(self)}")

    def automated_diff(self, **values):
        r"""
        Method to compute automated differentiation of a Sparse polynomial

        This method uses the idea of Automatic Differentiation to compute
        using an evaluation with `n`-ual numbers (see class:`clue.nual.NualNumber`)
        the evaluation of the polynomial together with all the evaluations of
        the partial derivatives of the polynomial.

        This method only works if the provided values covers all possible variables
        of the polynomial.

        Input:
            ``values`` - dictionary with ``(varname, value)`` for (at least) all the variables
            appearing in ``self`` (see method ``variables``).

        Output:
            A tuple `(p_0,...,p_k)` where `p_0` is the evaluation of ``self`` at the given point,
            and `p_i` is the evaluation of the partial derivative of ``self`` at the given point
            with respect to the `i`-th variable of ``self.gens``.

        Examples::

            TODO add the examples
        """
        if any(v not in values for v in self.variables()):
            raise ValueError(
                "Not enough information provided for automatic differentiation"
            )

        gens = self._varnames
        n = len(gens)

        if self.is_constant():
            return NualNumber([self.domain.convert(self.ct)] + [0 for _ in range(n)])
        to_eval = {
            gens[i]: NualNumber(
                [values.get(gens[i], 0)] + [1 if j == i else 0 for j in range(n)]
            )
            for i in range(n)
        }

        result = self.eval(**to_eval).ct
        if not isinstance(result, NualNumber):  # evaluation was zero
            return NualNumber((n + 1) * [self.domain.convert(0)])
        return result

    # --------------------------------------------------------------------------

    def exp(self, power):
        """
        Exponentiation, ``power`` is a *positive* integer
        """
        if power < 0:
            raise ValueError(f"Cannot raise to power {power}, {str(self)}")
        if power == 0:
            return SparsePolynomial.from_const(1, self._varnames, self.domain)
        if power == 1:
            return self
        if power % 2 == 0:
            return self.exp(power // 2) * self.exp(power // 2)
        return self * self.exp(power // 2) * self.exp(power // 2)

    # --------------------------------------------------------------------------

    def is_zero(self):
        r"""
        Checks equality with `0`.

        This methods checks whether a :class:`SparsePolynomial` is exactly 0 or not.

        Output
            ``True`` if ``self == 0`` holds.

        Examples::

            >>> from clue.rational_function import *
            >>> sp = SparsePolynomial.from_string("1",['x'])
            >>> sp.is_zero()
            False
            >>> sp = SparsePolynomial(['x'])
            >>> sp.is_zero()
            True
            >>> sp = SparsePolynomial.from_string("x**2*y - 2*x*y", ['x','y','z'])
            >>> sp.is_zero()
            False
        """
        if len(self._data) == 0:
            return True
        return False

    def is_unitary(self):
        r"""
        Checks equality with `1`.

        This methods checks whether a :class:`SparsePolynomial` is exactly 1 or not.

        Output
            ``True`` if ``self == 1`` holds.

        Examples::

            >>> from clue.rational_function import *
            >>> sp = SparsePolynomial.from_string("1",['x'])
            >>> sp.is_unitary()
            True
            >>> sp = SparsePolynomial(['x'])
            >>> sp.is_unitary()
            False
            >>> sp = SparsePolynomial.from_string("x**2*y - 2*x*y", ['x','y','z'])
            >>> sp.is_unitary()
            False
        """
        if self._data == {(): 1}:
            return True
        return False

    def is_constant(self):
        r"""
        Checks whether a polynomial is a constant

        This method checks whether a :class:`SparsePolynomial` is a constant or not. For doing so
        we simply check if ``self`` is zero (see :func:`is_zero`) or if the only monomial in the
        polynomial is `()` (i.e., the monomial `1`).

        Output
            ``True`` if ``self`` is a constant polynomial, ``False`` otherwise.

        Examples::

            TODO: add examples
        """
        return self.is_zero() or (len(self._data) == 1 and () in self._data)

    def is_linear(self):
        return all(
            (monomial == () or (len(monomial) == 1 and monomial[0][1] == 1))
            for monomial in self._data
        )

    # --------------------------------------------------------------------------

    def _pair_to_str(self, pair):
        if pair[1] == 1:
            return self._varnames[pair[0]]
        else:
            return f"{self._varnames[pair[0]]}**{pair[1]}"

    # --------------------------------------------------------------------------

    def _scalar_to_str(self, c):
        # not an elegant way to force elements of algebraic fields be printed with sqrt
        if isinstance(c, sympy.polys.polyclasses.ANP):
            dummy_ring = sympy.ring([], self.domain)[0]
            return f"({dummy_ring(c).as_expr()})"
        if isinstance(c, sympy.polys.fields.FracElement):
            return f"({c})"
        return str(c)

    # --------------------------------------------------------------------------

    def _monom_to_str(self, m, c):
        if c == 0:
            return "+", "0"

        prefix = "+"
        try:
            if c < 0:
                c *= -1
                prefix = "-"
        except:
            pass

        if not m:
            return prefix, self._scalar_to_str(c)

        # at this moment the coefficient is positive (or not comparable to 0)
        return prefix, (
            "" if c == self.domain.one else self._scalar_to_str(c) + "*"
        ) + "*".join(map(lambda p: self._pair_to_str(p), m))

    # --------------------------------------------------------------------------

    def __repr__(self):
        if not self._data:
            return "0"
        # at least one term is included in the polynomial
        terms = [self._monom_to_str(m, c) for m, c in self._data.items()]

        return (
            (terms[0][0] if terms[0][0] == "-" else "")
            + terms[0][1]
            + (" " if len(terms) > 1 else "")
            + " ".join(" ".join(term) for term in terms[1:])
        )

    # --------------------------------------------------------------------------

    def linear_part_as_vec(self) -> SparseVector:
        out = SparseVector(len(self._varnames), self.domain)
        for i in range(len(self._varnames)):
            if ((i, 1),) in self._data:
                out[i] = self._data[((i, 1),)]
        return out

    # --------------------------------------------------------------------------

    def get_sympy_dict(self):
        result = dict()
        for monom, coef in self._data.items():
            new_monom = [0] * len(self._varnames)
            for var, exp in monom:
                new_monom[var] = exp
            result[tuple(new_monom)] = coef
        return result

    # --------------------------------------------------------------------------

    def get_constant(self):
        return self.constant_term

    def get_sympy_ring(self):
        return sympy.polys.rings.ring(self._varnames, self.domain)[0]

    def to_sympy(self):
        r"""
        Returns the SymPy polynomial represented by ``self``.

        All the elements of type :class:`SparsePolynomial` can be transformed into
        an element in a SymPy polynomial ring. This is useful for some functionalities
        (see :func:`lcm` and :func:`gcd`).

        This method is simply a natural sequence of the methods :func:`get_sympy_ring`
        and :func:`get_sympy_dict`.

        Output
            A SymPy polynomial represented by ``self``.

        Examples::

            >>> from clue.rational_function import *
            >>> sp = SparsePolynomial.from_string("x**2*y-x*z**2", ['x','y','z'])
            >>> type(sp.to_sympy())
            <class 'sympy.polys.rings.PolyElement'>
            >>> sp.to_sympy()
            x**2*y - x*z**2
            >>> SparsePolynomial.from_sympy(sp.to_sympy()) == sp
            True
        """
        return self.get_sympy_ring()(self.get_sympy_dict())

    def change_base(self, new_domain):
        r"""Change the domain of the SparsePolynomial and creates a copy for it"""
        return SparsePolynomial(self._varnames, new_domain, self._data, True)

    # --------------------------------------------------------------------------
    def derivative(self, var_name):
        """
        Returns derivative of polynomial with respect to var_name
        """
        if var_name in self._varnames:
            var = self._varnames.index(var_name)
        else:
            return 0

        data = dict()
        for monom, coeff in self._data.items():
            for i in range(len(monom)):
                v, exp = monom[i]
                if v == var:
                    if exp == 1:
                        m_der = tuple(list(monom[:i]) + list(monom[(i + 1) :]))
                    else:
                        m_der = tuple(
                            list(monom[:i]) + [(var, exp - 1)] + list(monom[(i + 1) :])
                        )
                    data[m_der] = coeff * exp

        return SparsePolynomial(self._varnames, domain=self._domain, data=data)

    # --------------------------------------------------------------------------

    @staticmethod
    def lcm(polys):
        r"""
        Returns lowest common multiple of given polynomials (computed w/ SymPy)

        This method computes (using SymPy) the least common multiple of a list of
        sparse polynomials. This method assumes that all the :class:`SparsePolynomials`
        generates the same SymPy ring (see method :func:`get_sympy_ring`) and
        can be casted naturally to it.

        Input
            ``polys`` - list of sparse polynomials to compute the least common multiple.

        Output
            A :class:`SparsePolynomial` with the least common multiple of all the
            polynomials in ``polys``.

        Examples::

            >>> from clue.rational_function import *
            >>> sp1 = SparsePolynomial.from_string("x*y**2 + x**2*y", ['x','y'])
            >>> sp2 = SparsePolynomial.from_string("x**2*y**2", ['x','y'])
            >>> lcm = SparsePolynomial.lcm([sp1,sp2])
            >>> lcm == SparsePolynomial.from_string("x**2*y**3 + x**3*y**2", ['x','y'])
            True
        """
        R = polys[0].get_sympy_ring()
        sympy_polys = [R(poly.get_sympy_dict()) for poly in polys]
        result = sympy_polys[0]
        for p in sympy_polys[1:]:
            result = result.lcm(p)
        return SparsePolynomial.from_sympy(result)

    @staticmethod
    def gcd(polys):
        """
        Returns greatest common divisor of given polynomials (computed w/ SymPy)
        """
        polys_sp = [p.to_sympy() for p in polys]
        result = polys_sp[0]
        for p in polys_sp[1:]:
            result = result.gcd(p)
        return SparsePolynomial.from_sympy(result)

    # --------------------------------------------------------------------------

    @staticmethod
    def from_sympy(sympy_poly, varnames=None):
        r"""Static method inverse to :func:`to_sympy`"""
        domain = sympy_poly.ring.domain
        # lambda used to handle the case of the algebraic field of coefficients
        if varnames is None:
            varnames = list(map(lambda g: str(g.as_expr()), sympy_poly.ring.gens))
        data = dict()
        sympy_dict = sympy_poly.to_dict()
        for monom, coef in sympy_dict.items():
            new_monom = []
            for i in range(len(monom)):
                if monom[i]:
                    new_monom.append((i, monom[i]))
            data[tuple(new_monom)] = coef
        return SparsePolynomial(varnames, domain, data)

    @staticmethod
    def from_vector(vector, varnames=None, domain=QQ):
        r"""Static method inverse to :func:`linear_part_as_vec`"""
        from .linalg import SparseVector

        if isinstance(vector, SparseVector):
            if len(varnames) != vector.dim:
                raise TypeError(
                    f"The list must have as many elements ({len(vector)}) as variables ({len(varnames)})"
                )
            domain = vector.field
            data = {((i, 1),): vector[i] for i in vector.nonzero}
        else:
            if len(vector) != len(varnames):
                raise TypeError(
                    f"The list must have as many elements ({len(vector)}) as variables ({len(varnames)})"
                )
            data = {((i, 1),): el for i, el in enumerate(list) if el != 0}
        return SparsePolynomial(varnames, domain, data)

    # --------------------------------------------------------------------------

    @staticmethod
    def monomial(monomial, varnames, domain):
        each_var = []
        for pair in monomial:
            each_var += [
                SparsePolynomial.var_from_string(varnames[pair[0]], varnames, domain)
                ** pair[1]
            ]
        result = SparsePolynomial.from_const(1, varnames, domain)
        for el in each_var:
            result *= el
        return result

    @staticmethod
    def var_from_string(vname, varnames, domain=QQ):
        i = varnames.index(vname)
        return SparsePolynomial(varnames, domain, {((i, 1),): domain.one})

    @staticmethod
    def from_const(c, varnames, domain=QQ):
        return SparsePolynomial(varnames, domain, {tuple(): domain.convert(c)})

    @staticmethod
    def from_string(s, varnames, domain=QQ):
        return RationalFunction.from_string(s, varnames, domain).get_poly()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class RationalFunction:
    r"""
    Class for representing rational function with sparse polynomials

    Input:
        ``num`` - numerator SparsePolynomial
        ``denom`` - denominator SparsePolynomial

    Examples::

        >>> from clue.rational_function import *
        >>> from sympy import parse_expr, simplify
        >>> rat_fun = "0.1"
        >>> parsed = parse_expr(rat_fun)
        >>> rf = RationalFunction.from_string(rat_fun, ["a", "b", "c"])
        >>> simplify(parse_expr(str(rf)) - parsed) == 0
        True
        >>> rat_fun = "1 / a + 1 / b"
        >>> parsed = parse_expr(rat_fun)
        >>> rf = RationalFunction.from_string(rat_fun, ["a", "b", "c"])
        >>> simplify(parse_expr(str(rf)) - parsed) == 0
        True
        >>> rat_fun = "1 / (1 + 1/b + 1/(c + 1 / (a + b + 1/c)))"
        >>> parsed = parse_expr(rat_fun)
        >>> rf = RationalFunction.from_string(rat_fun, ["a", "b", "c"])
        >>> simplify(parse_expr(str(rf)) - parsed) == 0
        True
        >>> rat_fun = "(a + b) / (1 - a + 1/ (b + c)) - 3/5 + (7 + a) / (c + 1 / b)"
        >>> parsed = parse_expr(rat_fun)
        >>> rf = RationalFunction.from_string(rat_fun, ["a", "b", "c"])
        >>> simplify(parse_expr(str(rf)) - parsed) == 0
        True
        >>> rat_fun = "(a + b + c**2)**5 - 3 * a + b * 17 * 19 * 0.5"
        >>> parsed = parse_expr(rat_fun)
        >>> rf = RationalFunction.from_string(rat_fun, ["a", "b", "c"])
        >>> simplify(parse_expr(str(rf)) - parsed) == 0
        True
    """

    __parser = None
    __parser_stack = []

    def __init__(self, numer, denom):
        ## Checking the input has the correct format
        assert isinstance(numer, SparsePolynomial)
        assert isinstance(denom, SparsePolynomial)
        assert numer.domain == denom.domain
        assert numer.gens == denom.gens
        assert not denom.is_zero()

        ## Assigning the values for the rational function
        self._domain = numer.domain
        self._varnames = numer.gens
        self.numer = numer
        self.denom = (
            denom
            if not self.numer.is_zero()
            else SparsePolynomial.from_const(1, self.numer._varnames, self._domain)
        )

        ## Simplifying the rational function if the denominator is not 1
        if self._domain.is_Exact and denom != SparsePolynomial.from_const(
            1, self.gens, self.domain
        ):
            self.simplify()

    @staticmethod
    def from_const(val, varnames, domain=QQ):
        return RationalFunction(
            SparsePolynomial.from_const(val, varnames, domain),
            SparsePolynomial.from_const(1, varnames, domain),
        )

    # --------------------------------------------------------------------------
    def is_polynomial(self):
        return self.denom.is_constant()

    def get_poly(self):
        if self.is_polynomial():
            return self.numer * (
                self.domain.convert(1) / self.domain.convert(self.denom.ct)
            )
        raise ValueError(f"{self} is not a polynomial")

    def is_zero(self):
        return self.numer.is_zero()

    def is_constant(self):
        return self.is_zero() or (self.numer.is_constant() and self.denom.is_constant())

    # --------------------------------------------------------------------------
    @property
    def domain(self):
        return self._domain

    # --------------------------------------------------------------------------
    @property
    def gens(self):
        return self._varnames.copy()

    # --------------------------------------------------------------------------
    @property
    def size(self):
        return self.denom.size + self.numer.size

    def variables(self, as_poly=False):
        return tuple(
            set(
                list(self.numer.variables(as_poly))
                + list(self.denom.variables(as_poly))
            )
        )

    # --------------------------------------------------------------------------
    def valuation(self, var_name):
        r"""
        Valuation of a rational function w.r.t. a variable.

        A valuation is a map `\nu: D \rightarrow \mathbb{Z}` such that

        .. MATH::

            nu(p\cdot q) = \nu(p) + \nu(q) \qquad \nu(p + q) \geq \min(\nu(p), \nu(q))

        In particular, the functions `\nu_v(p/q) = \deg_v(p) - \deg_v(q)` is such a
        valuation. This method returns the valuation w.r.t. a variable of this
        rational function. It is based on the method :func:`clue.rational_function.SparsePolynomial.degree`.

        Input
            ``var_name`` - name (string) of the variable to compute the degree.

        Output
            The valuation of ``self`` w.r.t. ```var_name``.

        TODO: add examples and tests
        """
        return self.numer.degree(var_name) - self.denom.degree(var_name)

    # --------------------------------------------------------------------------
    def derivative(self, var):
        r"""
        Compute the derivative with respect to a given variable.

        This method computes the derivative of the rational function represented by ``self``
        with respect to a variable provided by ``var``.

        A rational function `f(x) = p(x)/q(x)` always satisfies the quotient rule for derivations:

        .. MATH::

            f'(x) = \frac{p'(x)q(x) - q'(x)p(x)}{q(x)^2}

        This method uses such formula and the method :func:`~clue.rational_function.SparsePolynomial.derivative`.

        Input
            ``var`` - name (string) of the variable with respect we compute the derivative.

        Output
            A rational function :class:`RationalFunction` with the derivative of ``self`` w.r.t. ``var``.

        Examples::

            >>> from clue.rational_function import *
            >>> varnames = ['x','y','z']
            >>> rf1 = RationalFunction.from_string("(3 * x**2 * y**4 * z**7)/(7*x**4 + 3*y**2 * z**9)", varnames)
            >>> rf1dx_expected = RationalFunction.from_string("(-(6*y**4*z**7*x*(7*x**4-3*y**2*z**9)))/((7*x**4+3*y**2*z**9)**2)", varnames)
            >>> rf1.derivative('x') == rf1dx_expected
            True
            >>> rf2 = RationalFunction.from_string("(x**2*y**2)/(z**2)", varnames)
            >>> rf2dx_expected = RationalFunction.from_string("(2*y**2*x)/(z**2)", varnames)
            >>> rf2.derivative('x') == rf2dx_expected
            True
            >>> rf2dz_expected = RationalFunction.from_string("(-(2*x**2*y**2))/(z**3)", varnames)
            >>> rf2.derivative('z') == rf2dz_expected
            True
            >>> rf3 = RationalFunction.from_string("(x**2)/(y*z**2)", varnames)
            >>> rf3dx_expected = RationalFunction.from_string("(2*x)/(y*z**2)", varnames)
            >>> rf3.derivative('x') == rf3dx_expected
            True
            >>> rf3dy_expected = RationalFunction.from_string("-(x**2)/(y**2*z**2)", varnames)
            >>> rf3.derivative('y') == rf3dy_expected
            True
            >>> rf3dz_expected = RationalFunction.from_string("(-2*x**2)/(y*z**3)", varnames)
            >>> rf3.derivative('z') == rf3dz_expected
            True

        If the variable provided does not show up in the rational function, the zero function is returned::

            >>> rf1.derivative('a')
            RationalFunction(0, 1)
            >>> rf1.derivative('a') == 0
            True
            >>> rf1.derivative('xy') == 0
            True
            >>> rf = RationalFunction.from_string("(x)/(2 * y**2)", varnames)
            >>> rf_dz = rf.derivative('z')
            >>> print(rf_dz)
            (0)/(1)
        """
        d_num = self.denom * self.numer.derivative(
            var
        ) - self.numer * self.denom.derivative(var)
        d_denom = self.denom * self.denom
        return RationalFunction(d_num, d_denom)

    # --------------------------------------------------------------------------
    def simplify(self):
        r"""
        Simplify a rational function in-place.

        Method that removes the common factors between the numerator and
        denominator of ``self``. It is based on the method :func:`~clue.rational_function.SparsePolynomial`
        and the exact division implementation.

        The simplification is performed *in-place*, meaning there is no output for this method, but
        instead, the result is stored within the same object.
        """
        # Removing the gcd of numerator and denominator (whatever Sympy finds)
        gcd = SparsePolynomial.gcd([self.numer, self.denom])
        if not gcd.is_unitary():
            self.numer = self.numer // gcd
            self.denom = self.denom // gcd

        # Removing the content of the denominator
        c = SparsePolynomial.from_const(self.denom.content, self.gens, self.domain)
        if not c.is_unitary():
            self.numer = self.numer // c
            self.denom = self.denom // c

    # --------------------------------------------------------------------------
    def __str__(self):
        return f"({self.numer})/({self.denom})"

    def __repr__(self):
        return f"RationalFunction({self.numer}, {self.denom})"

    # --------------------------------------------------------------------------
    def __mul__(self, other):
        if type(other) == RationalFunction:
            rf = RationalFunction(self.numer * other.numer, self.denom * other.denom)
        else:
            rf = RationalFunction(self.numer * other, self.denom)
        return rf

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if type(other) == RationalFunction:
            if self.denom == other.denom:
                rf = RationalFunction(self.numer + other.numer, self.denom)
            else:
                rf = RationalFunction(
                    self.numer * other.denom + other.numer * self.denom,
                    self.denom * other.denom,
                )
            return rf
        elif type(other) == SparsePolynomial:
            return self + RationalFunction(
                other, SparsePolynomial.from_const(1, self.gens, self.domain)
            )
        else:
            return self + RationalFunction.from_const(other, self.gens, self.domain)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return RationalFunction(self.numer * other.denom, self.denom * other.numer)

    # --------------------------------------------------------------------------

    def __neg__(self):
        return RationalFunction(-self.numer, self.denom)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __isub__(self, other):
        self += -other
        return self

    def __iadd__(self, other):
        if self.denom == other.denom:
            self.numer += other.numer
            return self
        self = self + other
        return self

    # --------------------------------------------------------------------------
    def eval(self, **values):
        r"""
        Method that evaluates a rational function.

        This method evaluates a rational function performing a simultaneous substitution of the
        given variables for some specific values. This is based on the method
        :func:`~clue.rational_function.SparsePolynomial.eval`. See that method for further
        limitations.

        Input
            values - dictionary containing the names fo the variables to be evaluated and the values to plug-in.

        Output
            the evaluation in the given values.

        Examples::

            >>> from clue.rational_function import *
            >>> rf = RationalFunction.from_string("x/(y*z**2)", ['x','y','z'])
            >>> print(rf.eval(x=0))
            0
            >>> print(rf.eval(y=1,z=2))
            1/4*x
            >>> print(rf.eval(y=2))
            (1/2*x)/(z**2)

        The denominator can not be evaluated to zero, or an :class:`ZeroDivisionError` would be raised::

            >>> print(rf.eval(y=0))
            Traceback (most recent call last):
            ...
            ZeroDivisionError: A zero from the denominator was found
        """
        # we evaluate first the denominator
        denom = self.denom.eval(**values)

        if denom.is_zero():  # if zero: raise an error
            raise ZeroDivisionError("A zero from the denominator was found")

        # otherwise we evaluate numerator and compute the quotient
        numer = self.numer.eval(**values)
        return numer / denom

    def subs(self, to_subs=None, **values):
        r"""
        Method to substitute variables in a rational function (not only with points)

        See method :func:`clue.rational_functions.SparsePolynomial.subs` for further information.
        """
        denom = self.denom.subs(to_subs, **values)

        if denom == 0:
            raise ZeroDivisionError(
                "A zero from the denominator was found while substituting"
            )

        numer = self.numer.subs(to_subs, **values)
        return numer / denom

    @cached_property
    def numerical_evaluator(self):
        return eval(
            f"lambda {','.join(self._varnames)}: ({str(self.numer)})/({str(self.denom)})"
        )

    def automated_diff(self, **values):
        return self.numer.automated_diff(**values) / self.denom.automated_diff(**values)

    # --------------------------------------------------------------------------
    def get_constant(self):
        return self.numer.ct / self.denom.ct

    def linear_part_as_vec(self) -> SparseVector:
        constant_parts = self.automated_diff(**{v: 0 for v in self.gens})
        out = SparseVector(len(self.gens), self.domain)
        for i in range(len(self.gens)):
            out[i] = constant_parts[i + 1]
        return out

    def get_sympy_ring(self):
        return sympy.polys.rings.ring(self.gens, self.domain)[0]

    def change_base(self, new_domain):
        r"""Change the domain of the RationalFunction"""
        return RationalFunction(
            self.numer.change_base(new_domain), self.denom.change_base(new_domain)
        )

    # --------------------------------------------------------------------------
    def __eq__(self, other):
        r"""
        Equality method for :class:`RationalFunction`.

        Two rational functions `p(x)/q(x)` and `r(x)/s(x)` are equal if and only if

        .. MATH::

            p(x)s(x) - q(x)r(x) = 0.

        This method checks such identity for ``self`` and ``other``. In case that ``other``
        is not a :class:`RationalFunction`, the method :func:`RationalFunction.from_string`
        is used to try and convert ``other`` into a rational function.

        Since we need to check and identity of polynomials, this method is based on
        :func:`clue.rational_function.SparsePolynomial.__eq__`.

        Input
            ``other`` - object to compare with ``self``.

        Output
            ``True`` if ``other`` and ``self`` are mathematically equal, ``False`` otherwise.

        Examples::

            >>> from clue.rational_function import *
            >>> rf1 = RationalFunction.from_string("x/y",['x','y'])
            >>> rf2 = RationalFunction.from_string("x/y",['x','y'])
            >>> rf1 is rf2
            False
            >>> rf1 == rf2
            True
        """
        if not isinstance(other, type(self)):
            if not isinstance(other, RationalFunction):
                try:
                    other = RationalFunction.from_string(
                        str(other), self.gens, self.domain
                    )
                except (ParseException, TypeError):
                    return NotImplemented
        return self.numer * other.denom == other.numer * self.denom

    def __hash__(self):
        return hash(self.numer) * hash(self.denom)

    # --------------------------------------------------------------------------
    def exp(self, power):
        """
        Exponentiation, ``power`` is a *positive* integer
        """
        if power < 0:
            raise ValueError(f"Cannot raise to power {power}, {str(self)}")
        if power == 1:
            return self
        if power % 2 == 0:
            return self.exp(power // 2) * self.exp(power // 2)
        return self * self.exp(power // 2) * self.exp(power // 2)

    # --------------------------------------------------------------------------
    @staticmethod
    def from_string(s, varnames, domain=QQ, var_to_ind=None):
        """
        Parsing a string to a polynomial, string is allowed to include floating-point numbers
        in the standard and scientific notation, they will be converted to rationals

        The code is an adapted version of fourFn example for pyparsing library by Paul McGuire
        https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py
        """

        def push_first(toks):
            RationalFunction.__parser_stack.append(toks[0])

        def push_unary_minus(toks):
            for t in toks:
                if t == "-":
                    RationalFunction.__parser_stack.append("unary -")
                else:
                    break

        # Creating a parser instance if necessary
        if RationalFunction.__parser is None:
            fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
            ident = Regex(
                f"(\\d[{alphanums+'_$'}]*[{alphas}]+[{alphanums+'_$'}]*)|([{alphas}]+[{alphanums+'_$'}]*)"
            )  # Word(alphanums, alphanums + "_$") # ident = Word(alphas, alphanums + "_$")
            plus, minus, mult, div = map(Literal, "+-*/")
            lpar, rpar = map(Suppress, "()")
            addop = plus | minus
            multop = mult | div
            expop = Literal("^") | Literal("**")

            expr = Forward()
            atom = (
                addop[...]
                + (
                    (fnumber | ident).setParseAction(
                        push_first
                    )  # (ident | fnumber).setParseAction(push_first)
                    | Group(lpar + expr + rpar)
                )
            ).setParseAction(push_unary_minus)

            factor = Forward()
            factor <<= atom + (expop + factor).setParseAction(push_first)[...]
            term = factor + (multop + factor).setParseAction(push_first)[...]
            expr <<= term + (addop + term).setParseAction(push_first)[...]
            RationalFunction.__parser = expr

        # parsing
        try:
            RationalFunction.__parser.parseString(s, parseAll=True)
        except:
            print(s)
            raise

        # for fast lookup
        var_ind_map = (
            {v: i for i, v in enumerate(varnames)} if var_to_ind is None else var_to_ind
        )

        def evaluate_stack(s):
            op = s.pop()
            if op == "unary -":
                return -evaluate_stack(s)
            if op in "+-*/":
                # note: operands are pushed onto the stack in reverse order
                op2 = evaluate_stack(s)
                op1 = evaluate_stack(s)
                if op == "+":
                    if op1.size < op2.size:
                        op1, op2 = op2, op1
                    op1 += op2
                    return op1
                if op == "-":
                    op1 -= op2
                    return op1
                if op == "*":
                    return op1 * op2
                if op == "/":
                    return op1 / op2
            if op == "^" or op == "**":
                exp_str = s.pop()
                exp = to_rational(exp_str)
                if exp.denominator != 1:
                    raise ValueError(
                        "invalid literal for int() with base 10: %s" % exp_str
                    )
                exp = int(exp)
                base = evaluate_stack(s)
                return base.exp(exp)
            if re.match(r"^[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?$", op):
                return RationalFunction.from_const(to_rational(op), varnames, domain)
            return RationalFunction(
                SparsePolynomial(
                    varnames, domain, {((var_ind_map[op], 1),): domain.one}
                ),
                SparsePolynomial.from_const(1, varnames, domain),
            )

        return evaluate_stack(RationalFunction.__parser_stack)

    @staticmethod
    def from_sympy(sympy_expr, varnames, domain=QQ):
        num, den = sympy_expr.as_expr().as_numer_denom()
        num = SparsePolynomial.from_string(str(num), varnames, domain)
        den = SparsePolynomial.from_string(str(den), varnames, domain)
        return RationalFunction(num, den)


# ------------------------------------------------------------------------------

__all__ = ["SparsePolynomial", "RationalFunction"]
