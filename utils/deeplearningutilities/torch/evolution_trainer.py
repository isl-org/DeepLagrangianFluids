from .trainer import Trainer, STOP_TIME
from .my_checkpoint_management import MyCheckpointManager
import signal


class EvolutionTrainer(Trainer):

    def __init__(
        self,
        root_dir,
        evolutions,
        keep_checkpoint_steps,
        save_interval_minutes=30,
        signal_handler_signals=(signal.SIGINT, signal.SIGUSR1, signal.SIGTERM),
    ):
        """
        Creates a new EvolutionTrainer object.
        This will create the root_dir for training and a directory for 
        checkpoints and log files within this directory.

        In contrast to the default Trainer, EvolutionTrainer supports evolutions.
        Evolutions allow to implement trainings with multiple stages that need to run
        the training script multiple times.
        In addition EvolutionTrainer takes care of checkpoint management.

        root_dir               : str
                                 path to the root of the training directory.

        evolutions             : list of objects
                                 The evolution object must implement the attributes 'name'
                                 and 'stop_step'.
                                 'name' is used in the filename of the checkpoint.
                                 'stop_step' is the iteration number at which the evolution
                                 stops.

        keep_checkpoint_steps  : list or set of integers
                                 A set of steps for which checkpoints are kept.

        save_interval_minutes : int
                                The save interval in minutes. If the latest checkpoint is
                                older than this interval then a new checkpoint will be
                                created by save_if_needed()

        signal_handler_signals : list of signals
                                 This object will install a signal handler for these signals
                                 that will cause keep_training() to return False and create
                                 a checkpoint.
        """
        if not evolutions:
            raise ValueError("List of evolutions must not be empty")
        if len([e.stop_step for e in evolutions]) != len(
                set([e.stop_step for e in evolutions])):
            raise ValueError("Duplicate 'stop_step' found in evolutions")
        if len([e.name for e in evolutions]) != len(
                set([e.name for e in evolutions])):
            raise ValueError("Duplicate 'name' found in evolutions")

        super().__init__(root_dir, signal_handler_signals)
        self._evolutions = sorted(evolutions, key=lambda x: x.stop_step)
        self._current_evolution = self._get_current_evolution()

        if keep_checkpoint_steps:
            self._keep_checkpoint_steps = set(keep_checkpoint_steps)
        else:
            self._keep_checkpoint_steps = set()

        # make sure that we keep checkpoints for the last step of each evolution
        for evo in self._evolutions:
            self._keep_checkpoint_steps.add(evo.stop_step)

        self._save_interval_minutes = save_interval_minutes

        self._checkpoint_manager = None

    def checkpoint_prefix_for_evo(self, evo):
        return 'ckpt_{}'.format(evo.name)

    def _get_current_evolution(self):
        for evo in self._evolutions:
            checkpoint_prefix = self.checkpoint_prefix_for_evo(evo)
            ckpt_manager = MyCheckpointManager(
                None,
                self.checkpoint_dir, [],
                checkpoint_prefix=checkpoint_prefix)

            steps_checkpoints = ckpt_manager.get_steps_and_checkpoints()
            if steps_checkpoints:
                last_step = steps_checkpoints[-1][0]
                if last_step < evo.stop_step:
                    return evo  # select unfinished evo as the current one
            else:
                return evo  # select evo without checkpoints as the current one
        return evo  # return the last evo if all evos are finished

    @property
    def current_evolution(self):
        return self._current_evolution

    @property
    def latest_checkpoint(self):
        """Returns the latest checkpoint across evolutions"""
        checkpoint = None
        for evo in self._evolutions:
            checkpoint_prefix = self.checkpoint_prefix_for_evo(evo)
            ckpt_manager = MyCheckpointManager(
                None,
                self.checkpoint_dir, [],
                checkpoint_prefix=checkpoint_prefix)
            if ckpt_manager.latest_checkpoint:
                checkpoint = ckpt_manager.latest_checkpoint
            else:
                break

        return checkpoint

    def _get_checkpoint_manager(self, checkpoint_fn):
        if self._checkpoint_manager is None:
            checkpoint_prefix = self.checkpoint_prefix_for_evo(
                self.current_evolution)
            self._checkpoint_manager = MyCheckpointManager(
                checkpoint_fn,
                self.checkpoint_dir,
                self._keep_checkpoint_steps,
                self._save_interval_minutes,
                checkpoint_prefix,
            )
        return self._checkpoint_manager

    def keep_training(
        self,
        step_var,
        checkpoint_fn,
        stop_time=STOP_TIME,
        display_interval=10,
        display_str_list=None,
        runstats_interval_minutes=10,
        step_var_increment=1,
    ):
        """
        This function increments the step_var, displays and logs runtime information and saves checkpoints.
        The function is intended to be used as the condition for the training loop, e.g.

        trainer = EvolutionTrainer(train_dir)

        step_var = torch.tensor(0)
        checkpoint_fn = lambda step: {'step': step_var', 'model': model.state_dict()}

        if trainer.latest_checkpoint:
            checkpoint = torch.load(trainer.latest_checkpoint)
            step_var = checkpoint['step']
            model.load_state_dict(checkpoint['model'])
        
        while trainer.keep_training(step_var, checkpoint_fn):
            train()



        step_var           : Scalar torch Tensor.
                             The step variable that will be incremented each call.

        checkpoint_fn      : A function returning the dictionary to be saved.

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
        status = super().keep_training(
            step_var=step_var,
            stop_step=self.current_evolution.stop_step,
            checkpoint_manager=self._get_checkpoint_manager(checkpoint_fn),
            stop_time=stop_time,
            display_interval=display_interval,
            display_str_list=display_str_list,
            runstats_interval_minutes=runstats_interval_minutes,
            step_var_increment=step_var_increment,
        )

        return status
