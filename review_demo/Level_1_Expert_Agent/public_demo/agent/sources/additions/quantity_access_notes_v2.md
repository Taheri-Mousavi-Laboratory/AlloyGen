# Accessing documented thermodynamic quantities

Author-written general guidance, replacing the earlier author note that prohibited
short quantity strings. Source: [TC-Python 2024a API guide](https://thermocalc.com/wp-content/uploads/Documentation/Archive/2020-2024/2024a/2024a-tc-python-api-programmer-guide.pdf),
ThermodynamicQuantity factory description, section 5.5.

Prefer named quantity factory methods where suitable. The factory covers common
quantities, not every available quantity. Documented Console Mode quantity strings
are also valid in the corresponding API methods. Verify a quantity's definition,
scope and units rather than inventing a factory method or guessing a symbol.
