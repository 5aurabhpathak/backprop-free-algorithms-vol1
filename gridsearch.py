"""
MIT license notice
© 2024 Saurabh Pathak. All rights reserved
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Purpose: subclass keras tuner gridsearch to make trial_id random instead of sequential as well as to save the training
metrics of all models build and trained. It resolves visualization issues
with several runs in tensorboard with conflicting trial numbers and allows plotting model metrics with matplotlib using
saved histories of all models. The code in this module is identical to that in keras_tuner.oracles.GridSearchOracle
except for a few bookkeeping related changes.
"""
import copy
import os

import keras_tuner
import numpy as np

from keras_tuner.src.engine import trial as trial_module, tuner as tuner_module, tuner_utils


class GridSearchOracle(keras_tuner.oracles.GridSearchOracle):
    def create_trial(self, tuner_id):
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        # Record all running client Tuner IDs.
        self.tuner_ids.add(tuner_id)

        # Pick the Trials waiting for retry first.
        if len(self._retry_queue) > 0:
            trial = self.trials[self._retry_queue.pop()]
            trial.status = trial_module.TrialStatus.RUNNING
            self.ongoing_trials[tuner_id] = trial
            self.save()
            self._display.on_trial_begin(trial)
            return trial

        # this subclass was needed for just this change. Rest all code is same as keras-tuner's
        trial_id = str(np.random.randint(1000000))

        if self.max_trials and len(self.trials) >= self.max_trials:
            status = trial_module.TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response["status"]
            values = response["values"] if "values" in response else None

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}

        trial = trial_module.Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == trial_module.TrialStatus.RUNNING:
            # Record the populated values (active only). Only record when the
            # status is RUNNING. If other status, the trial will not run, the
            # values are discarded and should not be recorded, in which case,
            # the trial_id may appear again in the future.
            self._record_values(trial)

            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self.start_order.append(trial_id)
            self._save_trial(trial)
            self.save()
            self._display.on_trial_begin(trial)

        # Remove the client Tuner ID when triggered the client to exit
        if status == trial_module.TrialStatus.STOPPED:
            self.tuner_ids.remove(tuner_id)

        return trial


class GridSearch(tuner_module.Tuner):
    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        save_best_model=False,
        **kwargs,
    ):
        self.seed = seed
        oracle = GridSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        self.save_best_model = save_best_model
        self.trial_counter = 0

        # to save histories of all models built and trained by this tuner
        self.histories = []

        super().__init__(oracle, hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        # Not using `ModelCheckpoint` to support MultiObjective.
        # It can only track one of the metrics to save the best model.
        if self.save_best_model:
            model_checkpoint = tuner_utils.SaveBestEpoch(
                objective=self.oracle.objective,
                filepath=self._get_checkpoint_fname(trial.trial_id),
            )
        original_callbacks = kwargs.pop("callbacks", [])
        self.trial_counter += 1

        # Run the training process multiple times.
        histories = []
        status_filename = os.path.join(self.get_trial_dir(trial.trial_id), 'status.txt')
        for execution in range(self.executions_per_trial):
            with open(status_filename, 'w') as f:
                f.write(f'trial number: {self.trial_counter} | running execution: {execution}')

            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))

            # Only checkpoint the best epoch across all executions.
            if self.save_best_model:
                callbacks.append(model_checkpoint)

            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)

            histories.append(obj_value)

        os.remove(status_filename)
        self.histories.append(dict(hp=trial.hyperparameters, histories=histories))
        return histories
