from eplus import pyEpError


def decode_packet_simple(packet):
    # Returns a list of float outputs from E+
    comp = packet.split(" ")
    comp = comp[:-1]
    comp_values = [float(s) for s in comp]
    output = []
    if comp_values[0] == 2:  # Version 2
        if comp_values[1] == 0:  # Simulation still running
            num_real = int(comp_values[2])
            time = comp_values[5]
            reals = comp_values[6:6 + num_real]
            output = reals
        else:
            switch = {
                1: "Simulation Finished. No output",
                -10: "Initialization Error",
                -20: "Time Integration Error",
                -1: "An Unspecified Error Occured",
            }
            print(switch.get(comp_values[1]))
    else:
        raise pyEpError.VersionError
    return output


def encode_packet(setpoints, time):
    # Takes in a list of lists with the real, int, and boolean values to input
    comp = [2, 0,
            len(setpoints[0]), len(setpoints[1]), len(setpoints[2]),
            time]
    for i in range(0, 3):
        comp.extend(setpoints[i])
    str_comp = [str(val) for val in comp]
    str_comp.extend("\n")
    output = " ".join(str_comp)
    return output


def decode_packet(packet):
    # Takes in a packet from ep_process.read() and returns a list of lists
    # corresponding to the real, int, and boolean values
    # Returns an empty list if there are no more outputs,
    # or if an error occured
    comp = packet.split(" ")
    comp = comp[:-1]
    comp_values = [float(s) for s in comp]
    output = []
    if comp_values[0] == 2:  # Version 2
        if comp_values[1] == 0:  # Simulation still running
            num_real = int(comp_values[2])
            num_int = int(comp_values[3])
            num_bool = int(comp_values[4])
            time = comp_values[5]

            reals = comp_values[6:6 + num_real]
            ints = [int(comp_values[i]) for i in range(6 + num_real,
                                                       6 + num_real + num_int)]
            bools = [
                comp_values[i] == 1
                for i in range(6 + num_real + num_int,
                               6 + num_real + num_int + num_bool)
            ]
            output.append(reals)
            output.append(ints)
            output.append(bools)
        else:
            switch = {
                1: "Simulation Finished. No output",
                -10: "Initialization Error",
                -20: "Time Integration Error",
                -1: "An Unspecified Error Occured",
            }
            print(switch.get(comp_values[1]))
    else:
        raise pyEpError.VersionError
    return output


def encode_packet_simple(setpoints, time):
    """Encodes all setpoints as reals to input to energyplus

    Args:
        setpoints: list of setpoints values
        time: number of ts.
    """
    comp = [2, 0, len(setpoints), 0, 0, time]
    comp.extend(setpoints)

    str_comp = [str(val) for val in comp]
    str_comp.extend("\n")
    output = " ".join(str_comp)
    return output
