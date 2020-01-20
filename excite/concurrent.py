"""
    @Author Jay Lee
    My attempt to provide a set of generic functions
    that enable ease of leverage of the power that
    concurrency brings to programs.
    TODO: Work on this during my downtime
"""
import multiprocessing as mp


def parallel_process(handler_fn, *args, process_count=5):
    """
        A function that causes a feature to run in parallel.
        Note that it is highly recommended to not add
        :return:
    """
    outputs = mp.Queue()
    processes = [mp.Process(target=handler_fn, args=(outputs, args)) for i in range(process_count)]

    for process in processes:
        process.start()

    for p in processes:
        p.join()

    results = [outputs.get() for process in processes]

    return results


def process_in_parallel(process_count=5, tasks=[]):
    """
        General-purpose function for programming
        :param process_count: The number of processes used
        for the given task
        :return:
    """
    outputs = mp.Queue()

    def parallel_process_decorator(process_fn):
        """
            :param process_fn: The function that is being "decorated"
            :return:
        """
        def output_fn(*args, **kwargs):

            def handler_fn(task, task_no):
                print(f"task: {task}. Task no: {task_no}")
                output = process_fn(task, task_no, *args, **kwargs)
                outputs.put(output)
                return output

            processes = [mp.Process(target=handler_fn, args=(tasks[i], i)) for i in range(len(tasks))]

            for process in processes:
                process.start()

            # Exit completed processes
            for p in processes:
                p.join()

            # Store results in list
            results = [outputs.get() for process in processes]

            return results

        return output_fn

    return parallel_process_decorator


if __name__ == "__main__":
    import random
    import string

    tasks = [1, 2, 3, 4, 5, 6]


    @process_in_parallel(process_count=7, tasks=tasks)
    def double_number(task, task_no):
        print(f"Handling task: {task}. Process no: {task_no}")
        return task * 2

    test = double_number()
    print(f"Final output: {test}")
