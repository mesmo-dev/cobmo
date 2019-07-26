"""Script template.

- This script is a template
- It prints "Hello world!"
"""

import os.path  # Import like this, instead of `from os import path` or `from os.path import *`
import pandas as pd  # Use abbreviations only for very common packages

import cobmo.config  # Imports from current package should appear last.


def template_function(
    hello_string,
    world_string="world"
):
    """This is where the action happens.

    - Extended description.
    """

    # Multi-line operations look like this. Always use brackets for multi-line operations!
    hello_world_string = (
        hello_string
        + " "
        + world_string
        + "!"
    )

    something_else_string = "Something else."

    return (
        hello_world_string,
        something_else_string
    )


def main():
    """This is the main entry point to your script."""

    (
        hello_world_string,
        something_else_string
    ) = template_function(
        hello_string="Hello"
    )

    print(hello_world_string)


if __name__ == '__main__':
    main()
