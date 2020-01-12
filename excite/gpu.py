"""
    @author Jay Lee
    Logic related to GPU.
    Some features include parsing the content of nvidia-smi to get
    information on GPU usage.
"""
import re


def parse_nvidia_smi_output(nvidia_smi_output):
    """
        Parse the output of "nvidia-smi" command to get gpu info.
        As nvidia-smi information gets updated, this code will also need to change
        :param nvidia_smi_output: the output of the "nvidia-smi" command
        :return: output with the f
    """
    # First 6 rows are junk so discard
    # Each row has a length of 80 characters in total
    row_index = 0
    data_interval = 3
    result = []
    for row in nvidia_smi_output[7:]:
        # First row has gpu index and name. Grab those
        if row_index % data_interval == 0:
            stripped_row = row.strip()
            if len(stripped_row) == 0:
                break
            current_gpu_dict = {}

            # GPU index
            gpu_index = re.search(r'\d+', row).group()
            current_gpu_dict["index"] = int(gpu_index)

        # This row contains memory consumption and total available memory
        elif row_index % data_interval == 1:
            row_split = row.split("|")
            # Add fan temp and row_split
            fan_temp = row_split[1].split("   ")[1]
            current_gpu_dict['fan_temp'] = fan_temp

            # Memory info
            current_consumption, total_consumption = re.findall(r'\d+', row_split[2])
            current_gpu_dict['current_memory'] = int(current_consumption)
            current_gpu_dict['total_memory'] = int(total_consumption)
        else:
            # Every third row is junk
            result.append(current_gpu_dict)

        row_index += 1

    return result



