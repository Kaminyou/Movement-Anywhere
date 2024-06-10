from ..utils import summarize_gait_parameters


def test_summarize_gait_parameters():
    # Prepare input data for the test
    test_input = [
        {'sl': 1.0, 'sw': 0.8, 'st': 0.5, 'v': 1.2, 'c': 0.2},
        {'sl': 1.1, 'sw': 0.7, 'st': 0.6, 'v': 1.3, 'c': 0.1},
        {'sl': 0.9, 'sw': 0.9, 'st': 0.4, 'v': 1.1, 'c': 0.3}
    ]

    # Expected output calculated manually
    expected_output = {
        'sl': np.mean([1.0, 1.1, 0.9]),
        'sw': np.mean([0.8, 0.7, 0.9]),
        'st': np.mean([0.5, 0.6, 0.4]),
        'v': np.mean([1.2, 1.3, 1.1]),
        'c': np.mean([0.2, 0.1, 0.3])
    }

    # Run the function with the test input
    result = summarize_gait_parameters(test_input)

    # Assert that the output from the function matches the expected output
    assert result == expected_output, "The summarized parameters do not match expected values."
