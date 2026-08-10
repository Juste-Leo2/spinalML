def pytest_addoption(parser):
    parser.addoption(
        "--debug-math", 
        action="store_true", 
        default=False, 
        help="Generate true_math_errors.log with detailed precision errors"
    )
