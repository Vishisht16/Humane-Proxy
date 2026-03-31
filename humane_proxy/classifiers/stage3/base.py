# Copyright 2026 Vishisht Mishra (Vishisht16)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract base class for Stage-3 reasoning classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from humane_proxy.classifiers.models import ClassificationResult


class Stage3Classifier(ABC):
    """Interface that all Stage-3 providers must implement.

    Stage-3 classifiers are **async** because they call external APIs
    (LlamaGuard, OpenAI, etc.).  They receive the prior pipeline result
    so they can skip work if the prior stages already reached a clear
    conclusion.
    """

    @abstractmethod
    async def classify(
        self, text: str, prior: ClassificationResult
    ) -> ClassificationResult:
        """Run the Stage-3 classification on *text*.

        Parameters
        ----------
        text:
            The raw user message.
        prior:
            The best result from Stages 1+2, for context.

        Returns
        -------
        ClassificationResult
            A result with ``stage=3``.
        """
        ...
