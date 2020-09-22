import os
import datetime
import resource
import signal
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ..runstats import *


def _is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def _get_stop_time(time_buffer=5 * 60):
    """Retrieves the stop time from the environment variable STOP_TIME.

    time_buffer: int
                 Time buffer in seconds. Default is 5 min to have enough time for the
                 shutdown.

    Returns None if the variable has not been set.
    """
    if 'STOP_TIME' in os.environ:
        try:
            stop_time = int(os.environ['STOP_TIME']) - time_buffer
            return stop_time
        except:
            pass
    return None


STOP_TIME = _get_stop_time()


class Trainer:

    def __init__(
        self,
        root_dir,
        signal_handler_signals=(signal.SIGINT, signal.SIGUSR1, signal.SIGTERM),
    ):
        """
        Creates a new Trainer object.
        This will create the root_dir for training and a directory for 
        checkpoints and log files within this directory.

        root_dir               : str
                                 path to the root of the training directory.

        signal_handler_signals : list of signals
                                 This object will install a signal handler for these signals
                                 that will cause keep_training() to return False and create
                                 a checkpoint.
        """
        self._root_dir = root_dir
        self._log_dir = os.path.join(root_dir, 'logs')
        self._checkpoint_dir = os.path.join(root_dir, 'checkpoints')

        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._summary_writer = None

        self._current_step = None
        self._keep_training_call_count = 0
        self._start_time = None
        self._start_step = None
        self._summary_iteration_timer = None
        self._display_iteration_timer = None
        self._last_runstats_time = None
        self._cpu_load = None
        self._gpu_accounting = None

        self._true_every_n_minutes_last_time = {}

        # setup signal handler to stop the training loop
        self._stop_signal_received = False

        def signal_handler(signum, frame):
            print("== received signal {} ==".format(signum), flush=True)
            self._stop_signal_received = True

        for sig in signal_handler_signals:
            signal.signal(sig, signal_handler)

    @property
    def STATUS_TRAINING_FINISHED(self):
        return 0

    @property
    def STATUS_TRAINING_UNFINISHED(self):
        return 100

    @property
    def STATUS_TRAINING_ERROR(self):
        return 1

    @property
    def stop_signal_received(self):
        return self._stop_signal_received

    @property
    def summary_writer(self):
        """The summary writer used by this Trainer object"""
        if self._summary_writer is None:
            self._summary_writer = SummaryWriter(self._log_dir)
        return self._summary_writer

    @property
    def checkpoint_dir(self):
        """Path to the checkpoint directory"""
        return self._checkpoint_dir

    @property
    def current_step(self):
        """The current step as int. 
        Note that the actual step variable has already been increased for the 
        next iteration by keep_training() and is current_step+1.
        """
        return self._current_step

    def _true_every_n_minutes(self, n, name):
        now = time.time()

        key = (n, name)
        if not key in self._true_every_n_minutes_last_time:
            self._true_every_n_minutes_last_time[key] = now
            return True
        else:
            last = self._true_every_n_minutes_last_time[key]
            if now - last > 60 * n:
                self._true_every_n_minutes_last_time[key] = now
                return True

        return False

    def log_scalar_every_n_minutes(self, n, name, value):
        """Convenience function for calling summary_writer.add_scalar in regular time intervals."""
        if self._true_every_n_minutes(n, name):
            self.summary_writer.add_scalar(name, value, self.current_step)

    def keep_training(
        self,
        step_var,
        stop_step,
        checkpoint_manager,
        stop_time=STOP_TIME,
        display_interval=10,
        display_str_list=None,
        runstats_interval_minutes=10,
        step_var_increment=1,
    ):
        """
        This function increments the step_var, displays and logs runtime information and saves checkpoints.
        The function is intended to be used as the condition for the training loop, e.g.

        trainer = Trainer(train_dir)

        checkpoint_manager = MyCheckpointManager(checkpoint, trainer.checkpoint_dir)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        
        while trainer.keep_training(step_var, stop_step=1000, checkpoint_manager=manager):
            train()


        step_var           : Scalar torch Tensor.
                             The step variable that will be incremented each call.

        stop_step         : int or list of int
                            One or more step values for which to stop the training.
                            On the first call to keep_training() the training will only be stopped for
                            the largest value in stop_step.
                            This mechanism can be used to design training procedures with multiple stages 
                            that need to call the training script multiple times.

        checkpoint_manager : object
                             CheckpointManager that implements a save_if_needed(step) function.
                             'save_if_needed' will be called every time.

        stop_time          : float or None
                             stop_time in seconds since the epoch. The default will be read from the
                             environment variable STOP_TIME.
                             Set to None for no stop_time or do not set the env var STOP_TIME.

        display_interval   : int
                             The interval in iterations for displaying runtime information on the console.

        display_str_list   : list
                             A list of additional objects that will be displayed with print().

        runstats_interval_minutes: int
                                   The interval for logging runtime statistics in minutes.
        
        step_var_increment : int
                             The value to add to the step_var. If 0 the step_var will not be updated.
        
        For a single stop_step the return behavior is
            Returns True if step_var != stop_step
            Returns False if step_var == stop_step or the current time is larger than stop_time
                          or if a signal has been received for which a signal handler was installed.
        """
        if not (isinstance(step_var, torch.Tensor) and
                len(step_var.shape) == 0):
            raise Exception(
                'step_var must be a scalar torch.Tensor, i.e., created with torch.tensor()'
            )
        self._keep_training_call_count += 1
        current_step = int(step_var)
        self._current_step = current_step

        now = time.time()

        max_step = np.max(stop_step)

        if (current_step == max_step or (isinstance(stop_step,
                                                    (list, tuple)) and
                                         self._keep_training_call_count > 1 and
                                         current_step in stop_step) or
            (stop_time and now > stop_time) or self.stop_signal_received):
            # Do not write a checkpoint if this is the first call to this
            # function. There could be checkpoint objects which are not yet
            # initialized at this point. Writing a checkpoint could overwrite
            # an existing checkpoint.
            if self._keep_training_call_count > 1:
                checkpoint_manager.save(current_step)
            return False

        if checkpoint_manager and not self._start_step is None:
            checkpoint_manager.save_if_needed(current_step)

        if self._start_step is None:
            self._summary_iteration_timer = IterationTimer()
            self._display_iteration_timer = IterationTimer()
            self._last_runstats_time = now
            self._cpu_load = CPULoad()
            self._gpu_accounting = GPUAccounting()
            self._start_time = now
            self._start_step = current_step

        # runstats summaries. write summaries more frequent in the beginning
        if (now - self._last_runstats_time > runstats_interval_minutes * 60 or
                _is_power_of_two(self._keep_training_call_count)
           ) and current_step > self._start_step:
            self._last_runstats_time = now

            # log some resource usage statistics and the current cpu/gpu load

            # log iterations per second
            time_per_iteration = self._summary_iteration_timer.get_avg_iteration_time(
                current_step)
            if time_per_iteration:
                self.summary_writer.add_scalar('runstats/iterPerSec',
                                               float(1 / time_per_iteration),
                                               current_step)

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self.summary_writer.add_scalar('runstats/maxrssMB',
                                           rusage.ru_maxrss // 2**10,
                                           current_step)
            self.summary_writer.add_scalar('runstats/swaps', rusage.ru_nswap,
                                           current_step)
            self.summary_writer.add_scalar('runstats/fileInputs',
                                           rusage.ru_inblock, current_step)
            self.summary_writer.add_scalar('runstats/fileOutputs',
                                           rusage.ru_oublock, current_step)
            self.summary_writer.add_scalar('runstats/pageFaults_minor',
                                           rusage.ru_minflt, current_step)
            self.summary_writer.add_scalar('runstats/pageFaults_major',
                                           rusage.ru_majflt, current_step)
            self.summary_writer.add_scalar('runstats/contextSwitches_voluntary',
                                           rusage.ru_nvcsw, current_step)
            self.summary_writer.add_scalar(
                'runstats/contextSwitches_involuntary', rusage.ru_nivcsw,
                current_step)

            # cpu stats
            cpu_times = self._cpu_load.get_avg_cpu_load()
            if cpu_times:
                avg_cpu_load = sum(cpu_times)
                self.summary_writer.add_scalar('runstats/cpuLoad', avg_cpu_load,
                                               current_step)
                self.summary_writer.add_scalar('runstats/cpuLoadUser',
                                               cpu_times[0], current_step)
                self.summary_writer.add_scalar('runstats/cpuLoadSys',
                                               cpu_times[1], current_step)

            # gpu stats
            gpu_stats = self._gpu_accounting.get_accounting_stats()
            if gpu_stats:
                for gpu_idx, stat in gpu_stats.items():
                    keys = ('gpuUtilization',
                           )  #'memoryUtilization', 'maxMemoryUsage')
                    for k in keys:
                        self.summary_writer.add_scalar(
                            'runstats/gpu{0}/{1}'.format(gpu_idx, k), stat[k],
                            current_step)

        # print display strings
        if display_interval and current_step % display_interval == 0:
            time_per_iteration = self._display_iteration_timer.get_avg_iteration_time(
                current_step)
            iterations_per_second_str = '{0:9.2f}'.format(
                1 / time_per_iteration) if time_per_iteration else 'n/a'
            if current_step > self._start_step:
                eta_iterations_per_second = (
                    current_step - self._start_step) / (now - self._start_time)
                remaining_secs = int(
                    (max_step - current_step) / eta_iterations_per_second)
                remaining_time_str = str(
                    datetime.timedelta(seconds=remaining_secs))
            else:
                remaining_time_str = 'n/a'

            print("# {0} {1:>8} {2} ips  {3:>18} rem | ".format(
                datetime.datetime.fromtimestamp(int(now)), current_step,
                iterations_per_second_str, remaining_time_str),
                  end='')
            if display_str_list:
                print(*display_str_list, flush=True)
            else:
                print('')

        # increment step variable
        if step_var_increment:
            step_var += step_var_increment

        return True
