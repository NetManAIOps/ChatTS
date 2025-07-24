# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference (RAGAS): https://github.com/explodinggradients/ragas

from evaluation.ragas.metric import AnswerCorrectness
from evaluation.ragas.config import load_llm, load_embeddings, config
from ragas import RunConfig
from loguru import logger
import copy

def calculate_ragas_score(question: str, response: str, label: str):
    try:
        answer_correctness = AnswerCorrectness(
            embeddings=load_embeddings(),
            llm=load_llm(),
            weights=[1.0, 0.0]
        )
        answer_correctness.answer_detail = {}

        score = answer_correctness.score(
            row={
                'question': question,
                'answer': response,
                'ground_truth': label
            }
        )
    except Exception as e:
        logger.error(f"Error calculating RAGAS score: {e}. Please make sure that you have set the correct API key!")
        return 0.0, {}

    return float(score), copy.deepcopy(answer_correctness.answer_detail)
