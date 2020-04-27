"""Config file that stores the paths to the SPlisHSPlasH binaries """
# Set this variable to the path of the DynamicBoundarySimulator binary
SIMULATOR_BIN = None

if SIMULATOR_BIN is None:
    raise ValueError(
        'Please set the path to the DynamicBoundarySimulator in {}'.format(
            __file__))


def _set_splishsplash_bin_paths(simulator_bin_path):
    """Check if the path points to the right binary.
    Throws an exception if the binary names do not match the expected names.
    On success sets the respective module vars to the expected paths.
    """
    import os
    bin_dir = os.path.dirname(simulator_bin_path)

    simulator_bin_name, extension = os.path.splitext(
        os.path.basename(simulator_bin_path))

    if 'DynamicBoundarySimulator' != simulator_bin_name:
        raise ValueError(
            "Wrong name for simulator binary, expected 'DynamicBoundarySimulator', got '{}'"
            .format(simulator_bin_path))

    volume_sampling_bin_path = os.path.join(bin_dir, 'VolumeSampling')
    if extension:
        volume_sampling_bin_path += '.' + extension

    if not os.path.isfile(volume_sampling_bin_path):
        raise FileNotFoundError(
            "Cannot find the VolumeSampling binary in the same dir as the simulator. Please check the path to the simulator."
        )

    global SIMULATOR_BIN, VOLUME_SAMPLING_BIN
    SIMULATOR_BIN = simulator_bin_path
    VOLUME_SAMPLING_BIN = volume_sampling_bin_path


VOLUME_SAMPLING_BIN = None  # will be derived from SIMULATOR_BIN by the function
_set_splishsplash_bin_paths(SIMULATOR_BIN)
