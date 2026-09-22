from __future__ import annotations

import random
from collections import Counter

from torch.utils.data import Sampler

from .unified_four_class_dataset import (
    POOL_FCLC_ANCHOR,
    POOL_LM_TARGET,
    POOL_ORDER,
    POOL_VV_TARGET,
    PatientBalancedCycle,
    UnifiedFourClassRecord,
)


STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS = {
    POOL_FCLC_ANCHOR: 5000,
    POOL_VV_TARGET: 2000,
    POOL_LM_TARGET: 1000,
}


class Stage3VirtualEpochSampler(Sampler[int]):
    """Patient-balanced 8000-exposure sampler with fixed-case exclusion."""

    def __init__(
        self,
        records: list[UnifiedFourClassRecord],
        *,
        steps_per_virtual_epoch: int = 1600,
        batch_size: int = 5,
        seed: int = 3,
        excluded_image_ids: set[str] | None = None,
    ):
        self.records = records
        self.steps_per_virtual_epoch = int(steps_per_virtual_epoch)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_size = self.steps_per_virtual_epoch * self.batch_size
        self.excluded_image_ids = set(excluded_image_ids or set())

        expected = sum(STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS.values())
        if self.epoch_size != expected:
            raise ValueError(
                f"epoch_size={self.epoch_size} must equal Stage3 exposure count {expected}"
            )
        self.by_pool = {
            pool: [
                index
                for index, record in enumerate(records)
                if record.pool_name == pool and record.image_id not in self.excluded_image_ids
            ]
            for pool in POOL_ORDER
        }
        for pool, values in self.by_pool.items():
            if not values:
                raise ValueError(f"Missing Stage3 training pool after fixed-case exclusion: {pool}")

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_cycles(self):
        base_seed = self.seed + self.epoch * 1000003
        return {
            pool: PatientBalancedCycle(self.records, self.by_pool[pool], base_seed + i * 7919)
            for i, pool in enumerate(POOL_ORDER)
        }

    def plan_indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch * 104729)
        tokens: list[str] = []
        for pool in POOL_ORDER:
            tokens.extend([pool] * STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS[pool])
        rng.shuffle(tokens)

        cycles = self._build_cycles()
        indices: list[int] = []
        for start in range(0, len(tokens), self.batch_size):
            used_images: set[str] = set()
            for pool in tokens[start : start + self.batch_size]:
                index = cycles[pool].next(used_images)
                image_id = self.records[index].image_id
                if image_id in self.excluded_image_ids:
                    raise RuntimeError(f"Fixed validation case entered Stage3 training: {image_id}")
                used_images.add(image_id)
                indices.append(index)

        pool_counts = Counter(self.records[index].pool_name for index in indices)
        if dict(pool_counts) != STAGE3_VIRTUAL_EPOCH_SAMPLE_COUNTS:
            raise RuntimeError(f"Stage3 exposure plan mismatch: {dict(pool_counts)}")
        return indices

    def __iter__(self):
        return iter(self.plan_indices())
