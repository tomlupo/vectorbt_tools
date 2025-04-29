# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.
#
# MIT License
#
# Copyright (c) 2018 Samuel Monnier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Classes for purged cross-validation in time series.

As described in Advances in Financial Machine Learning, Marcos Lopez de Prado, 2018."""

from abc import abstractmethod
from itertools import combinations

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks, datetime_ as dt

__all__ = [
    "PurgedWalkForwardCV",
    "PurgedKFoldCV",
]


class BasePurgedCV:
    """Abstract class for purged time series cross-validation.

    Time series cross-validation requires each sample has a prediction time, at which the features are used to
    predict the response, and an evaluation time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    `pred_times` and `eval_times` must all be pandas DataFrames/Series having the same index. It is also
    assumed that the samples are time-ordered with respect to the prediction time."""

    def __init__(self, n_folds: int = 10, purge_td: tp.TimedeltaLike = 0) -> None:
        self._n_folds = n_folds
        self._pred_times = None
        self._eval_times = None
        self._indices = None
        self._purge_td = dt.to_timedelta(purge_td)

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return self._n_folds

    @property
    def purge_td(self) -> pd.Timedelta:
        """Purge period."""
        return self._purge_td

    @property
    def pred_times(self) -> tp.Optional[pd.Series]:
        """Times at which predictions are made."""
        return self._pred_times

    @property
    def eval_times(self) -> tp.Optional[pd.Series]:
        """Times at which the response becomes available and the error can be computed."""
        return self._eval_times

    @property
    def indices(self) -> tp.Optional[tp.Array1d]:
        """Indices."""
        return self._indices

    def purge(
        self,
        train_indices: tp.Array1d,
        test_fold_start: int,
        test_fold_end: int,
    ) -> tp.Array1d:
        """Purge part of the train set.

        Given a left boundary index `test_fold_start` of the test set and a right boundary index
        `test_fold_end`, this method removes from the train set all the samples whose evaluation time
        is posterior to the prediction time of the first test sample after the boundary."""
        time_test_fold_start = self.pred_times.iloc[test_fold_start]
        eval_times = self.eval_times + self.purge_td
        train_indices_1 = np.intersect1d(train_indices, self.indices[eval_times < time_test_fold_start])
        train_indices_2 = np.intersect1d(train_indices, self.indices[test_fold_end:])
        return np.concatenate((train_indices_1, train_indices_2))

    @abstractmethod
    def split(
        self,
        X: tp.SeriesFrame,
        y: tp.Optional[tp.Series] = None,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
    ):
        """Yield the indices of the train and test sets."""
        checks.assert_instance_of(X, (pd.Series, pd.DataFrame), arg_name="X")
        if y is not None:
            checks.assert_instance_of(y, pd.Series, arg_name="y")
        if pred_times is None:
            pred_times = X.index
        if isinstance(pred_times, pd.Index):
            pred_times = pd.Series(pred_times, index=X.index)
        else:
            checks.assert_instance_of(pred_times, pd.Series, arg_name="pred_times")
            checks.assert_index_equal(X.index, pred_times.index, check_names=False)
        if eval_times is None:
            eval_times = X.index
        if isinstance(eval_times, pd.Index):
            eval_times = pd.Series(eval_times, index=X.index)
        else:
            checks.assert_instance_of(eval_times, pd.Series, arg_name="eval_times")
            checks.assert_index_equal(X.index, eval_times.index, check_names=False)

        self._pred_times = pred_times
        self._eval_times = eval_times
        self._indices = np.arange(X.shape[0])


class PurgedWalkForwardCV(BasePurgedCV):
    """Purged walk-forward cross-validation.

    The samples are decomposed into `n_folds` folds containing equal numbers of samples, without
    shuffling. In each cross validation round, `n_test_folds` contiguous folds are used as the
    test set, while the train set consists in between `min_train_folds` and `max_train_folds`
    immediately preceding folds.

    Each sample should be tagged with a prediction time and an evaluation time.
    The split is such that the intervals [`pred_times`, `eval_times`] associated to samples in
    the train and test set do not overlap. (The overlapping samples are dropped.)

    With `split_by_time=True` in the `PurgedWalkForwardCV.split` method, it is also possible to
    split the samples in folds spanning equal time intervals (using the prediction time as a time tag),
    instead of folds containing equal numbers of samples."""

    def __init__(
        self,
        n_folds: int = 10,
        n_test_folds: int = 1,
        min_train_folds: int = 2,
        max_train_folds: tp.Optional[int] = None,
        split_by_time: bool = False,
        purge_td: tp.TimedeltaLike = 0,
    ) -> None:
        BasePurgedCV.__init__(self, n_folds=n_folds, purge_td=purge_td)

        if n_test_folds >= self.n_folds - 1:
            raise ValueError("n_test_folds must be between 1 and n_folds - 1")
        self._n_test_folds = n_test_folds
        if min_train_folds >= self.n_folds - self.n_test_folds:
            raise ValueError("min_train_folds must be between 1 and n_folds - n_test_folds")
        self._min_train_folds = min_train_folds
        if max_train_folds is None:
            max_train_folds = self.n_folds - self.n_test_folds
        if max_train_folds > self.n_folds - self.n_test_folds:
            raise ValueError("max_train_split must be between 1 and n_folds - n_test_folds")
        self._max_train_folds = max_train_folds
        self._split_by_time = split_by_time
        self._fold_bounds = []

    @property
    def n_test_folds(self) -> int:
        """Number of folds used in the test set."""
        return self._n_test_folds

    @property
    def min_train_folds(self) -> int:
        """Minimal number of folds to be used in the train set."""
        return self._min_train_folds

    @property
    def max_train_folds(self) -> int:
        """Maximal number of folds to be used in the train set."""
        return self._max_train_folds

    @property
    def split_by_time(self) -> int:
        """Whether the folds span identical time intervals. Otherwise, the folds contain
        an (approximately) equal number of samples."""
        return self._split_by_time

    @property
    def fold_bounds(self) -> tp.List[int]:
        """Fold boundaries."""
        return self._fold_bounds

    def compute_fold_bounds(self) -> tp.List[int]:
        """Compute a list containing the fold (left) boundaries."""
        if self.split_by_time:
            full_time_span = self.pred_times.max() - self.pred_times.min()
            fold_time_span = full_time_span / self.n_folds
            fold_bounds_times = [self.pred_times.iloc[0] + fold_time_span * n for n in range(self.n_folds)]
            return self.pred_times.searchsorted(fold_bounds_times)
        else:
            return [fold[0] for fold in np.array_split(self.indices, self.n_folds)]

    def compute_train_set(self, fold_bound: int, count_folds: int) -> tp.Array1d:
        """Compute the position indices of the samples in the train set."""
        if count_folds > self.max_train_folds:
            start_train = self.fold_bounds[count_folds - self.max_train_folds]
        else:
            start_train = 0
        train_indices = np.arange(start_train, fold_bound)
        train_indices = self.purge(train_indices, fold_bound, self.indices[-1])
        return train_indices

    def compute_test_set(self, fold_bound: int, count_folds: int) -> tp.Array1d:
        """Compute the position indices of the samples in the test set."""
        if self.n_folds - count_folds > self.n_test_folds:
            end_test = self.fold_bounds[count_folds + self.n_test_folds]
        else:
            end_test = self.indices[-1] + 1
        return np.arange(fold_bound, end_test)

    def split(
        self,
        X: tp.SeriesFrame,
        y: tp.Optional[tp.Series] = None,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
    ) -> tp.Iterable[tp.Tuple[tp.Array1d, tp.Array1d]]:
        BasePurgedCV.split(self, X, y, pred_times=pred_times, eval_times=eval_times)

        self._fold_bounds = self.compute_fold_bounds()

        count_folds = 0
        for fold_bound in self.fold_bounds:
            if count_folds < self.min_train_folds:
                count_folds = count_folds + 1
                continue
            if self.n_folds - count_folds < self.n_test_folds:
                break
            test_indices = self.compute_test_set(fold_bound, count_folds)
            train_indices = self.compute_train_set(fold_bound, count_folds)

            count_folds = count_folds + 1
            yield train_indices, test_indices


class PurgedKFoldCV(BasePurgedCV):
    """Purged and embargoed combinatorial cross-validation.

    The samples are decomposed into `n_folds` folds containing equal numbers of samples, without shuffling.
    In each cross validation round, `n_test_folds` folds are used as the test set, while the other folds
    are used as the train set. There are as many rounds as `n_test_folds` folds among the `n_folds` folds.

    Each sample should be tagged with a prediction time and an evaluation time. The split is such
    that the intervals [`pred_times`, `eval_times`] associated to samples in the train and test set
    do not overlap. (The overlapping samples are dropped.) In addition, an "embargo" period is defined,
    giving the minimal time between an evaluation time in the test set and a prediction time in the
    training set. This is to avoid, in the presence of temporal correlation, a contamination of the
    test set by the train set."""

    def __init__(
        self,
        n_folds: int = 10,
        n_test_folds: int = 2,
        purge_td: tp.TimedeltaLike = 0,
        embargo_td: tp.TimedeltaLike = 0,
    ) -> None:
        BasePurgedCV.__init__(self, n_folds=n_folds, purge_td=purge_td)

        if n_test_folds > self.n_folds - 1:
            raise ValueError("n_test_folds must be between 1 and n_folds - 1")
        self._n_test_folds = n_test_folds
        self._embargo_td = dt.to_timedelta(embargo_td)

    @property
    def n_test_folds(self) -> int:
        """Number of folds used in the test set."""
        return self._n_test_folds

    @property
    def embargo_td(self) -> pd.Timedelta:
        """Embargo period."""
        return self._embargo_td

    def embargo(
        self,
        train_indices: tp.Array1d,
        test_indices: tp.Array1d,
        test_fold_end: int,
    ) -> tp.Array1d:
        """Apply the embargo procedure to part of the train set.

        This amounts to dropping the train set samples whose prediction time occurs within
        `PurgedKFoldCV.embargo_td` of the test set sample evaluation times. This method applies
        the embargo only to the part of the training set immediately following the end of the test
        set determined by `test_fold_end`."""
        last_test_eval_time = self.eval_times.iloc[test_indices[test_indices <= test_fold_end]].max()
        min_train_index = len(self.pred_times[self.pred_times <= last_test_eval_time + self.embargo_td])
        if min_train_index < self.indices.shape[0]:
            allowed_indices = np.concatenate((self.indices[:test_fold_end], self.indices[min_train_index:]))
            train_indices = np.intersect1d(train_indices, allowed_indices)
        return train_indices

    def compute_train_set(
        self,
        test_fold_bounds: tp.List[tp.Tuple[int, int]],
        test_indices: tp.Array1d,
    ) -> tp.Array1d:
        """Compute the position indices of the samples in the train set."""
        train_indices = np.setdiff1d(self.indices, test_indices)
        for test_fold_start, test_fold_end in test_fold_bounds:
            train_indices = self.purge(train_indices, test_fold_start, test_fold_end)
            train_indices = self.embargo(train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(
        self,
        fold_bound_list: tp.List[tp.Tuple[int, int]],
    ) -> tp.Tuple[tp.List[tp.Tuple[int, int]], tp.Array1d]:
        """Compute the position indices of the samples in the test set."""
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices

    def split(
        self,
        X: tp.SeriesFrame,
        y: tp.Optional[tp.Series] = None,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
    ) -> tp.Iterable[tp.Tuple[tp.Array1d, tp.Array1d]]:
        BasePurgedCV.split(self, X, y, pred_times=pred_times, eval_times=eval_times)

        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_folds)]
        selected_fold_bounds = list(combinations(fold_bounds, self.n_test_folds))
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices
