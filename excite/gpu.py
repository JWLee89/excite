"""
    @author Jay Lee
    Logic related to GPU.
    Some features include parsing the content of nvidia-smi to get
    information on GPU usage.
"""
import re
import sys


def parse_nvidia_smi_output(nvidia_smi_output, sort_key="index", reverse=False):
    """
        TODO: This needs to be optimized and rewritten later. Not very pretty.

        Parse the output of "nvidia-smi" command to get gpu info.
        As nvidia-smi information gets updated, this code will also need to change.
        Sorted in order
        :param nvidia_smi_output: the output of the "nvidia-smi" command
        :param sort_key: The key to sort by
        :param reverse: whether to sort in descending order
        :return: output with the f
    """
    # First 6 rows are junk so discard
    # Each row has a length of 80 characters in total
    row_index = 0
    data_interval = 3  # GPU info spans across three rows. Last row is junk.
    result = []
    total_free_memory = 0
    total_memory = 0
    total_consumed_memory = 0
    min_free_memory = sys.maxsize  # GPU index with free_est memory

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
            current_gpu_dict['free_memory'] = current_gpu_dict['total_memory'] - current_gpu_dict['current_memory']

            # Update statistics
            total_memory += current_gpu_dict['total_memory']
            total_consumed_memory += current_gpu_dict['current_memory']
            total_free_memory += current_gpu_dict['free_memory']
            min_free_memory = min(min_free_memory, current_gpu_dict['free_memory'])

        else:
            # Every third row is junk
            result.append(current_gpu_dict)

        row_index += 1

    return sorted(result, key=lambda i: i[sort_key], reverse=reverse), {'total_memory': total_memory,
                                                                        'total_free_memory': total_free_memory,
                                                                        'min_free_memory': min_free_memory,
                                                                        'total_consumed_memory': total_consumed_memory}
